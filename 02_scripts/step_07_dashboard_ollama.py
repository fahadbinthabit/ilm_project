# 02_scripts/step_07_dashboard_ollama.py
import os, time, json, traceback, requests
import gradio as gr
import pandas as pd

from step_01_extract import run_extraction
from step_02_chunk import run_chunking
from step_03_consolidate import run_consolidation
from step_04_generate_qa import run_qa_generation  # Ollama version

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

def run_step(step):
    try:
        t0 = time.time()
        if step == "extract": run_extraction()
        elif step == "chunk": run_chunking()
        elif step == "consolidate": run_consolidation()
        elif step == "generate_qa": run_qa_generation()
        elif step == "all":
            run_extraction(); run_chunking(); run_consolidation(); run_qa_generation()
        msg = f"✅ Step '{step}' finished in {time.time()-t0:.2f}s"
    except Exception as e:
        msg = "❌ " + "".join(traceback.format_exception_only(type(e), e)).strip()
    return msg, load_tables()

def load_tables():
    tabs = {}
    if os.path.exists("processing_status.csv"):
        tabs["Extraction Log"] = pd.read_csv("processing_status.csv")
    if os.path.exists("chunk_database.csv"):
        tabs["Chunk DB (preview)"] = pd.read_csv("chunk_database.csv").head(20)
    if os.path.exists("training_dataset.jsonl"):
        rows = []
        with open("training_dataset.jsonl","r",encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i>=10: break
                try: rows.append(pd.json_normalize([json.loads(line)]))
                except: pass
        if rows:
            tabs["Training Dataset (10 rows)"] = pd.concat(rows, ignore_index=True)
    return tabs

def chat_fn(user_msg, history):
    try:
        payload = {"model": OLLAMA_MODEL, "prompt": user_msg, "stream": False}
        r = requests.post(OLLAMA_URL, json=payload, timeout=600)
        r.raise_for_status()
        return r.json().get("response","").strip()
    except Exception as e:
        return f"[Ollama error] {e}"

with gr.Blocks(title="ILM Local (Ollama)") as app:
    gr.Markdown("# ILM Pipeline (Local CPU with Ollama)")
    with gr.Row():
        step = gr.Dropdown(choices=["all","extract","chunk","consolidate","generate_qa"], value="all", label="Pipeline Step")
        btn  = gr.Button("Run")
    log = gr.Markdown("Logs…")
    tabs = gr.Tabs()

    def render_tables(data):
        tabs.children = []
        for name, df in data.items():
            with gr.Tab(name):
                gr.Dataframe(df, interactive=False)

    btn.click(fn=run_step, inputs=step, outputs=[log, tabs])
    gr.Markdown("## Chat with your model (Ollama)")
    gr.ChatInterface(chat_fn, title="Insurance LLM (Local)")

app.launch()
