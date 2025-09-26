from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from io import BytesIO
import base64

from .matchers import ResumeRanker  # ✅ using ResumeRanker

app = FastAPI()
ranker = ResumeRanker(similarity_threshold=0.0, batch_size=8)  # allow all, sort later

class RankRequest(BaseModel):
    resume_texts: List[str]
    resume_filenames: List[str]
    resume_files_b64: Optional[List[str]] = None
    job_text: str
    job_name: str
    max_results: int = 5

@app.post("/rank-resumes")
async def rank_resumes(data: RankRequest):
    resume_texts = data.resume_texts
    resume_filenames = data.resume_filenames
    resume_files_b64 = data.resume_files_b64 or []

    # Fill missing texts from raw files if possible
    for i in range(len(resume_filenames)):
        if i >= len(resume_texts) or not resume_texts[i].strip():
            if i < len(resume_files_b64) and resume_files_b64[i]:
                file_bytes = base64.b64decode(resume_files_b64[i])
                text = ranker.extract_text_from_pdf(BytesIO(file_bytes))
                if i < len(resume_texts):
                    resume_texts[i] = text or ""
                else:
                    resume_texts.append(text or "")

    if len(resume_texts) != len(resume_filenames):
        raise HTTPException(status_code=400, detail="Mismatch after OCR extraction")

    results = ranker.rank_resumes(
        resume_texts, resume_filenames,
        data.job_text, data.job_name,
        max_keywords=15
    )

    return {"results": results}
