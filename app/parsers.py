import os
import re
import json
import pdfplumber
import logging
from typing import Dict
import pytesseract
from PIL import Image
import unicodedata

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import unicodedata

def clean_text(text: str) -> str:
    # Normalize unicode ligatures first
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", errors="ignore").decode()

    # Lowercase
    text = text.lower()

    # Remove known junk patterns
    junk_patterns = [
        r"\bcity\s*state\b", r"\bresume\b", r"\bcurriculum vitae\b", r"\bcv\b"
    ]
    for pattern in junk_patterns:
        text = re.sub(pattern, " ", text)

    # Remove standalone years
    text = re.sub(r"(?<![a-z])\b(19|20)\d{2}\b(?![a-z])", " ", text)

    # Remove month names unless part of a full date
    months = r"jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec"
    text = re.sub(rf"\b({months})\b", " ", text)

    # Keep programming terms
    text = re.sub(r"(?<!\w)(c\+\+|c#|\.net|node\.js|r|go)(?!\w)", r"\1", text)

    # Drop random short tokens (1–2 letters) except whitelist
    whitelist = {"c", "r", "go"}
    text = " ".join([tok for tok in text.split() if len(tok) > 2 or tok in whitelist])

    # Remove leftover punctuation
    text = re.sub(r"[^\w\s\.\+#]", " ", text)

    # Collapse spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using pdfplumber and fallback to OCR if needed."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ''
            for page in pdf.pages:
                # Extract text with advanced settings
                extracted = page.extract_text(
                    layout=True,
                    x_tolerance=2,
                    y_tolerance=2,
                    keep_blank_chars=True
                )
                if extracted:
                    text += extracted
                else:
                    # Fallback to OCR for scanned/image PDFs
                    img = page.to_image(resolution=300)
                    text += pytesseract.image_to_string(img.original)

            return clean_text(text.strip())
    except Exception as e:
        logger.error(f"Failed to extract text from {pdf_path}: {str(e)}")
        return ''

def batch_process_resumes(resume_dir: str, output_json: str, max_files: int = 2484) -> Dict[str, str]:
    """Batch process resume PDFs into a cleaned JSON file."""
    resumes = {}
    processed_count = 0

    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    for root, _, files in os.walk(resume_dir):
        for filename in files:
            if filename.endswith('.pdf') and processed_count < max_files:
                pdf_path = os.path.join(root, filename)
                relative_path = os.path.relpath(pdf_path, resume_dir).replace('\\', '/')
                text = extract_text_from_pdf(pdf_path)
                if text:
                    resumes[relative_path] = text
                    processed_count += 1
                    logger.info(f"Processed {relative_path} ({processed_count}/{max_files})")
                else:
                    logger.warning(f"No text extracted from {relative_path}")

    logger.info(f"Total resumes processed: {processed_count}")

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(resumes, f, indent=2)

    return resumes

if __name__ == '__main__':
    resume_dir = 'D:/resume-screener-mvp/data/resumes'
    output_json = 'D:/resume-screener-mvp/data/processed/extracted_texts.json'
    batch_process_resumes(resume_dir, output_json, max_files=2484)
