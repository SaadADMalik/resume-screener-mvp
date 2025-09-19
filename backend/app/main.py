from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import json
import PyPDF2
from parsers import extract_text_from_pdf
from matchers import match_resume_to_job

app = FastAPI()

@app.post('/match_resume')
async def match_resume(file: UploadFile = File(...), job: str = Form('data_scientist')):
    # Validate job description file
    job_file = f'D:/resume-screener-mvp/data/job_descriptions/{job}.txt'
    try:
        with open(job_file, 'r') as f:
            job_text = f.read()
    except FileNotFoundError:
        return JSONResponse(status_code=400, content={'error': f'Job description {job}.txt not found'})

    # Extract text from uploaded PDF
    try:
        with open('temp_resume.pdf', 'wb') as temp_file:
            temp_file.write(await file.read())
        resume_text = extract_text_from_pdf('temp_resume.pdf')
        if not resume_text:
            return JSONResponse(status_code=400, content={'error': 'Failed to extract text from PDF'})
    except Exception as e:
        return JSONResponse(status_code=400, content={'error': f'PDF processing failed: {str(e)}'})

    # Match resume to job
    result = match_resume_to_job([resume_text], ['uploaded_resume'], job_text)
    return result[0]

@app.post('/match_job')
async def match_job(job_text: str = Form(...), n: int = Form(10)):
    # Load extracted resume texts
    try:
        with open('D:/resume-screener-mvp/data/processed/extracted_texts.json', 'r') as f:
            resumes = json.load(f)
    except FileNotFoundError:
        return JSONResponse(status_code=404, content={'error': 'Resume data not found'})

    # Select up to 100 resumes for testing
    resume_filenames = list(resumes.keys())[:100]
    resume_texts = list(resumes.values())[:100]

    # Match and rank resumes
    results = match_resume_to_job(resume_texts, resume_filenames, job_text)
    return results[:n]

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)