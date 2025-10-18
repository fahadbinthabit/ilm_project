# ilm_project/02_scripts/step_04_generate_qa.py
# --- FINAL VERSION: Full Q&A generation using local Ollama ---

import os
import json
import time
import pandas as pd
from tqdm import tqdm
import requests  # For local Ollama calls

# --- Configuration ---
CHUNK_DATABASE_FILE = 'chunk_database.csv'
OUTPUT_DATASET_FILE = 'training_dataset.jsonl'
OLLAMA_MODEL_NAME = 'mistral'  # The model you downloaded with 'ollama pull'
OLLAMA_URL = "http://localhost:11434/api/generate"

# The prompt for generating Q&A pairs
QA_GENERATION_PROMPT = """
You are an insurance domain expert. Based ONLY on the following text context, generate one relevant and insightful Q&A pair.

**Context:**
"{context}"

**Instructions:**
1. The question should be something a professional in the insurance field might ask.
2. The answer must come directly from the provided context.
3. Output only valid JSON — no markdown, no commentary.

**JSON Output Format:**
{{
  "question": "Your question here.",
  "answer": "Your answer here."
}}
"""

def generate_qa_pair(context: str) -> dict | None:
    """Send one prompt to local Ollama and parse its JSON response."""
    payload = {
        "model": OLLAMA_MODEL_NAME,
        "prompt": QA_GENERATION_PROMPT.format(context=context),
        "format": "json",
        "stream": False,
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=600)
        r.raise_for_status()
        raw = r.json().get("response", "{}")
        qa = json.loads(raw)
        if qa.get("question") and qa.get("answer"):
            return qa
    except Exception as e:
        print(f"⚠️ Ollama error: {e}")
    return None


def run_qa_generation():
    """Main Q&A generation loop for all chunks."""
    try:
        df = pd.read_csv(CHUNK_DATABASE_FILE)
    except FileNotFoundError:
        print(f"❌ '{CHUNK_DATABASE_FILE}' not found. Run 'consolidate' first.")
        return

    total_chunks = len(df)
    print(f"🚀 Starting full Q&A generation for {total_chunks} chunks using model '{OLLAMA_MODEL_NAME}'.")

    with open(OUTPUT_DATASET_FILE, 'w', encoding='utf-8') as fout:
        for _, row in tqdm(df.iterrows(), total=total_chunks, desc="Step 4: Generating Q&A"):
            context = str(row.get('text', ''))[:4000]
            qa = generate_qa_pair(context)
            if qa:
                fout.write(json.dumps({
                    "instruction": qa["question"],
                    "context": context,
                    "response": qa["answer"]
                }, ensure_ascii=False) + '\n')

    print(f"🎉 Finished! Wrote dataset to '{OUTPUT_DATASET_FILE}'.")


if __name__ == "__main__":
    run_qa_generation()
