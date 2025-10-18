# 02_scripts/step_06_run_local_ollama.py
import requests, json

OLLAMA_MODEL = "mistral"  # or "llama3"
URL = "http://localhost:11434/api/generate"

print("💬 Local Ollama chat. Type 'exit' to quit.")
while True:
    msg = input("You: ")
    if msg.lower() in ("exit","quit"): break
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": msg,
        "stream": False,
    }
    r = requests.post(URL, json=payload, timeout=600)
    r.raise_for_status()
    print("ILM:", r.json().get("response","").strip(), "\n")
