# ilm_project/02_scripts/step_03_consolidate.py
import os
import json
import pandas as pd
from tqdm import tqdm

# --- Configuration ---
TRAINING_CORPUS_DIR = '01_data/training_corpus'
OUTPUT_CSV_FILE = 'chunk_database.csv' # Will be saved in the root project folder

def run_consolidation():
    """
    Reads all .jsonl chunk files and consolidates them into a single CSV.
    """
    jsonl_files = get_jsonl_files()
    if not jsonl_files:
        print("✅ No .jsonl files found to consolidate.")
        return

    print(f"🚀 Found {len(jsonl_files)} .jsonl file(s) to consolidate...")

    all_chunks = []
    for file_name in tqdm(jsonl_files, desc="Step 3: Consolidating Chunks"):
        file_path = os.path.join(TRAINING_CORPUS_DIR, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Load each line as a JSON object and add it to our list
                all_chunks.append(json.loads(line))

    if not all_chunks:
        print("⚠️ No chunks were found in the .jsonl files.")
        return

    # Use pandas to easily convert the list of dicts into a structured table
    # json_normalize is great for handling nested data like our 'metadata' field
    df = pd.json_normalize(all_chunks)

    # Save the consolidated data to a single CSV file
    df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8')

    print(f"\n🎉 Consolidation complete. {len(df)} chunks saved to '{OUTPUT_CSV_FILE}'.")


def get_jsonl_files():
    """Finds all .jsonl files in the training corpus directory."""
    if not os.path.exists(TRAINING_CORPUS_DIR):
        return []
    return [f for f in os.listdir(TRAINING_CORPUS_DIR) if f.lower().endswith('.jsonl')]


if __name__ == '__main__':
    run_consolidation()