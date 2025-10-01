from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import matchers
import logging

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic model for the request
class RankRequest(BaseModel):
    resume_texts: List[str]
    resume_filenames: List[str]
    resume_files_b64: Optional[List[str]] = None
    job_text: str
    job_name: str
    max_results: int = 5

# Pydantic model for the response
class RankResult(BaseModel):
    filename: str
    similarity: float
    kw_score: float
    impact_score: float
    combined_score: float
    matched_keywords: List[str]
    explanation: str

@app.post("/rank-resumes", response_model=List[RankResult])
async def rank_resumes(request: RankRequest):
    try:
        # Initialize ResumeRanker
        ranker = matchers.ResumeRanker(similarity_threshold=0.0)

        # Extract text from base64 if provided
        resume_texts = request.resume_texts
        if request.resume_files_b64:
            import base64
            from io import BytesIO
            resume_texts = [
                ranker.extract_text_from_pdf(BytesIO(base64.b64decode(b64)))
                for b64 in request.resume_files_b64
            ]
            print("Extracted text:", resume_texts)  # Debugging print

        # Validate lengths match
        if len(resume_texts) != len(request.resume_filenames):
            raise HTTPException(status_code=400, detail="Number of resume texts must match number of filenames")

        # Rank resumes using ResumeRanker
        results = ranker.rank_resumes(
            resume_texts=resume_texts,
            resume_filenames=request.resume_filenames,
            job_text=request.job_text,
            job_name=request.job_name,
            top_k=request.max_results
        )

        # Convert to RankResult objects
        ranked_results = [
            RankResult(
                filename=r["filename"],
                similarity=r["similarity"],
                kw_score=r["kw_score"],
                impact_score=r["impact_score"],
                combined_score=r["combined_score"],
                matched_keywords=r["matched_keywords"],
                explanation=r["explanation"]
            )
            for r in results
        ]

        return ranked_results

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)