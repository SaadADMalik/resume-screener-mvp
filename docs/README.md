# Resume Screener MVP
AI-powered resume screening project using NLP and FastAPI/Streamlit.

## Setup
- Virtual environment: env
- Backend dependencies: ackend/requirements.txt
- Frontend dependencies: rontend/requirements.txt

## Data
- Resumes: 2,484 PDFs in data/resumes/ (subset of [Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)), full dataset in Google Drive [insert_link_here]
- Job Descriptions: data/job_descriptions/data_scientist.txt, data/job_descriptions/web_developer.txt

### Parsing Notes
Processed 100 PDFs. Some may have failed due to encryption or image-based content. Used pdfplumber for better extraction if needed.
