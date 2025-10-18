# 02_scripts/step_07_dashboard.py
import os, io, sys, time, traceback
import gradio as gr
import pandas as pd

# Import pipeline steps
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from step_01_extract import run_extraction
from step_02_chunk import run_chunking
from step_03_consolidate import run_consolidation
# choose one generator implementation
GEN_IMPL = os.environ.get("ILM_QA_IMPL", "hf")  # "hf" or "ollama"
if GEN_IMPL == "ollama":
    from step_04_generate_qa import run_qa_generation
else:
    from step_04b_generate_qa_hf import run_qa_generation_hf as run_qa_generation

# Chat loading (base or fine-tuned)
def load_model():
    try:
        from unsloth import FastLanguageModel
        if os.path.isdir("insurance_llm"):
            name = "insurance_llm"
        else:
            name = "unsloth/llama-3-8b-Instruct-bnb-4bit"
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=name, max_seq_length=2048, dtype=None, load_in_4bit=True
        )
        return model, tokenizer, f"Loaded: {name}"
    except Exception as e:
        return None, None, f"Model load error: {e}"

MODEL, TOKENIZER, MODEL_STATUS = load_model()

def run_step(step):
    buf = io.StringIO()
    try:
        start = time.time()
        if step == "extract":
            run_extraction()
        elif step == "chunk":
            run_chunking()
        elif step == "consolidate":
            run_consolidation()
        elif step == "generate_qa":
            run_qa_generation()
        elif step == "all":
            run_extraction(); run_chunking(); run_consolidation(); run_qa_generation()
        msg = f"✅ Step '{step}' completed in {time.time()-start:.2f}s"
    except Exception as e:
        msg = "❌ " + "".join(traceback.format_exception_only(type(e), e)).strip()
    return msg, load_status_tables()

def load_status_tables():
    tables = {}
    if os.path.exists("processing_status.csv"):
        tables["Extraction Log"] = pd.read_csv("processing_status.csv")
    if os.path.exists("chunk_database.csv"):
        tables["Chunk DB (preview)"] = pd.read_csv("chunk_database.csv").head(20)
    if os.path.exists("training_dataset.jsonl"):
        # preview first 10 JSON lines
        rows = []
        with open("training_dataset.jsonl", "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 10: break
                rows.append(pd.json_normalize([eval(line)]))  # simple preview
        if rows:
            tables["Training Dataset (10 rows)"] = pd.concat(rows, ignore_index=True)
    return tables

def chat_fn(user_msg, history):
    if MODEL is None:
        return f"[{MODEL_STATUS}]"
    # Build a simple instruction formatting
    prompt = user_msg
    inputs = TOKENIZER(prompt, return_tensors="pt").to(MODEL.device)
    out = MODEL.generate(
        **inputs, max_new_tokens=512, temperature=0.7, top_p=0.9, do_sample=True
    )
    reply = TOKENIZER.decode(out[0], skip_special_tokens=True)
    return reply

with gr.Blocks(title="ILM Pipeline & Chat") as demo:
    gr.Markdown("# ILM Pipeline Dashboard")
    gr.Markdown(f"**Model status:** {MODEL_STATUS}")

    with gr.Row():
        step_dd = gr.Dropdown(
            choices=["all","extract","chunk","consolidate","generate_qa"],
            value="all", label="Pipeline Step"
        )
        run_btn = gr.Button("Run")

    log = gr.Markdown("Logs will appear here…")
    status_tabs = gr.Tabs()

    def render_tables(tables):
        # Clear and rebuild tabs
        status_tabs.children = []
        for name, df in tables.items():
            with gr.Tab(name):
                gr.Dataframe(df, interactive=False)

    run_btn.click(
        fn=run_step, inputs=step_dd, outputs=[log, status_tabs]
    )

    gr.Markdown("## Chat with your ILM")
    gr.ChatInterface(chat_fn, title="Insurance LLM")

demo.launch()
