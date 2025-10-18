# ilm_project/02_scripts/step_01_extract.py
import fitz, os, csv
from datetime import datetime
from tqdm import tqdm
# This line is now corrected
from utils import clean_and_normalize_text, is_header_or_footer

# --- PATHS CORRECTED ---
RAW_PDFS_DIR = '01_data/raw_pdfs'
PROCESSED_TEXT_DIR = '01_data/processed_text'
STATUS_LOG_FILE = 'processing_status.csv'

def run_extraction():
    os.makedirs(PROCESSED_TEXT_DIR, exist_ok=True)
    processed_files = get_processed_files()
    try:
        all_pdfs = [f for f in os.listdir(RAW_PDFS_DIR) if f.lower().endswith('.pdf')]
    except FileNotFoundError:
        print(f"❌ Error: Raw PDFs directory not found at '{RAW_PDFS_DIR}'. Please create it and add your PDFs.")
        return

    pdfs_to_process = [f for f in all_pdfs if f not in processed_files]
    if not pdfs_to_process:
        print("✅ No new PDFs to extract. Text files are up to date.")
        return

    print(f"🚀 Found {len(pdfs_to_process)} new PDF(s) to process...")
    for pdf_filename in tqdm(pdfs_to_process, desc="Step 1: Extracting Text"):
        pdf_path = os.path.join(RAW_PDFS_DIR, pdf_filename)
        try:
            text = extract_text_from_pdf(pdf_path)
            if text:
                out_path = os.path.join(PROCESSED_TEXT_DIR, f"{os.path.splitext(pdf_filename)[0]}.txt")
                with open(out_path, 'w', encoding='utf-8') as f: f.write(text)
                log_status(pdf_filename, 'success')
            else: raise ValueError("Extracted text was empty.")
        except Exception as e: log_status(pdf_filename, 'failed', str(e))
    print("\n🎉 Text extraction complete.")

def get_processed_files():
    processed = set()
    if not os.path.exists(STATUS_LOG_FILE): return processed
    with open(STATUS_LOG_FILE, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            next(reader, None) # Skip header
            for row in reader:
                if len(row) >= 2 and row[1] == 'success': processed.add(row[0])
        except StopIteration:
            pass 
    return processed

def log_status(filename, status, message=""):
    file_exists = os.path.exists(STATUS_LOG_FILE)
    with open(STATUS_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists or os.path.getsize(STATUS_LOG_FILE) == 0: writer.writerow(['filename', 'status', 'timestamp', 'message'])
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([filename, status, ts, message])

def extract_text_from_pdf(pdf_path: str) -> str:
    full_text = []
    doc = fitz.open(pdf_path)
    for page in doc:
        page_text = page.get_text("text")
        cleaned_page_text = [line for line in page_text.split('\n') if line.strip() and not is_header_or_footer(line, page.number + 1, doc.page_count)]
        full_text.append(" ".join(cleaned_page_text))
    return clean_and_normalize_text(" ".join(full_text))

if __name__ == '__main__':
    run_extraction()