# ilm_project/04_dashboard/app.py
from flask import Flask, render_template, request
import pandas as pd
import os

app = Flask(__name__)

# --- Configuration ---
CHUNK_DB_FILE = '../chunk_database.csv'
RAW_PDFS_DIR = '../01_data/raw_pdfs'

@app.route('/')
def dashboard():
    # Check if the chunk database exists
    if not os.path.exists(CHUNK_DB_FILE):
        return render_template('no_data.html')

    # Read the consolidated chunk data
    try:
        df = pd.read_csv(CHUNK_DB_FILE)
        # Handle cases where the CSV is empty
        if df.empty:
            return render_template('no_data.html')
    except pd.errors.EmptyDataError:
        return render_template('no_data.html')
    
    # --- Search Functionality ---
    search_query = request.args.get('search', '')
    if search_query:
        # Filter the DataFrame based on the search query (case-insensitive)
        df = df[df['text'].str.contains(search_query, case=False, na=False)]

    # Convert dataframe to a list of dictionaries for the template
    chunks = df.to_dict(orient='records')
    
    # --- Summary Stats ---
    total_chunks = len(chunks)
    total_docs = len(df['source_document'].unique()) if not df.empty else 0
    
    summary = {
        "total_chunks": total_chunks,
        "total_docs": total_docs,
        "search_query": search_query
    }
    
    return render_template('index.html', chunks=chunks, summary=summary)

if __name__ == '__main__':
    print("🚀 Starting ILM Dashboard...")
    print("🌍 Open your browser and go to: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)