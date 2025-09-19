import json
import re
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def extract_keywords(text):
    # Simple keyword extraction (extend with NLP if needed)
    keywords = re.findall(r'\b\w+\b', text.lower())
    return set([word for word in keywords if len(word) > 3])  # Filter short words

def match_resume_to_job(resume_texts, resume_filenames, job_text, model_name='all-MiniLM-L6-v2'):
    # Load model
    model = SentenceTransformer(model_name)
    
    # Embed texts in batch
    start_time = time.time()
    resume_embeddings = model.encode(resume_texts, batch_size=32, show_progress_bar=True)
    job_embedding = model.encode([job_text])[0]
    print(f'Embedding time: {time.time() - start_time:.2f} seconds')
    
    # Compute similarities
    similarities = cosine_similarity(resume_embeddings, [job_embedding])
    
    # Extract keywords and compile results
    results = {}
    job_keywords = extract_keywords(job_text)
    for filename, resume_text, similarity in zip(resume_filenames, resume_texts, similarities):
        resume_keywords = extract_keywords(resume_text)
        matched_keywords = list(resume_keywords.intersection(job_keywords))[:10]
        results[filename] = {
            'similarity': float(similarity[0]),
            'matched_keywords': matched_keywords
        }
    return results

if __name__ == '__main__':
    # Load extracted resume texts
    with open('D:/resume-screener-mvp/data/processed/extracted_texts.json', 'r') as f:
        resumes = json.load(f)
    
    # Select 100 resumes
    resume_filenames = list(resumes.keys())[:100]
    resume_texts = list(resumes.values())[:100]
    
    # Load job description
    with open('D:/resume-screener-mvp/data/job_descriptions/data_scientist.txt', 'r') as f:
        job_text = f.read()
    
    # Match resumes
    start_time = time.time()
    matches = match_resume_to_job(resume_texts, resume_filenames, job_text)
    print(f'Total processing time: {time.time() - start_time:.2f} seconds')
    
    # Save results
    with open('D:/resume-screener-mvp/data/processed/matches.json', 'w') as f:
        json.dump(matches, f, indent=2)
