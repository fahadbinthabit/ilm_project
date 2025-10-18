# ilm_project/02_scripts/utils.py
import re
from ftfy import fix_text
from unidecode import unidecode

def clean_and_normalize_text(text: str) -> str:
    text = fix_text(text)
    text = unidecode(text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'-\s+', '', text)
    return text

def is_header_or_footer(text: str, page_number: int, max_pages: int) -> bool:
    if str(page_number) in text: return True
    if len(text.split()) < 5: return True
    if "©" in text or "copyright" in text.lower(): return True
    return False