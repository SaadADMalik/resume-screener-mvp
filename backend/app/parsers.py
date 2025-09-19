import os
import PyPDF2
import json

def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + " "
            return text.strip()
    except Exception as e:
        with open("D:/resume-screener-mvp/data/processed/parsing_errors.log", "a") as log:
            log.write(f"Error processing {pdf_path}: {e}\n")
        return ""

def batch_process_resumes(resume_dir, output_json, max_files=100):
    resumes = {}
    for i, filename in enumerate(os.listdir(resume_dir)):
        if i >= max_files:
            break
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(resume_dir, filename)
            text = extract_text_from_pdf(pdf_path)
            if text:
                resumes[filename] = text
    with open(output_json, 'w') as f:
        json.dump(resumes, f, indent=2)

if __name__ == "__main__":
    resume_dir = "D:/resume-screener-mvp/data/resumes"
    output_json = "D:/resume-screener-mvp/data/processed/extracted_texts.json"
    batch_process_resumes(resume_dir, output_json)