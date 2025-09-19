import json
import re
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def extract_keywords(text):
    # Extract keywords and phrases (e.g., 'machine learning', 'data analysis')
    keywords = set()
    text = text.lower()
    # Simple regex for single words and common phrases
    single_words = re.findall(r'\b\w{4,}\b', text)  # Words with 4+ chars
    phrases = re.findall(r'\b\w+\s+\w+\b', text)  # Two-word phrases
    keywords.update(single_words)
    keywords.update(phrases)
    return keywords

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
    results = []
    job_keywords = extract_keywords(job_text)
    for filename, resume_text, similarity in zip(resume_filenames, resume_texts, similarities):
        resume_keywords = extract_keywords(resume_text)
        matched_keywords = list(resume_keywords.intersection(job_keywords))
        # Sort keywords by relevance (e.g., length, favoring phrases)
        matched_keywords = sorted(matched_keywords, key=len, reverse=True)[:5]  # Top 5
        results.append({
            'filename': filename,
            'similarity': float(similarity[0]),
            'matched_keywords': matched_keywords
        })
    
    # Sort by similarity (descending)
    results = sorted(results, key=lambda x: x['similarity'], reverse=True)
    return results

if __name__ == '__main__':
    # Load extracted resume texts
    with open('D:/resume-screener-mvp/data/processed/extracted_texts.json', 'r') as f:
        resumes = json.load(f)
    
    # Select up to 1000 resumes
    resume_filenames = list(resumes.keys())[:1000]
    resume_texts = list(resumes.values())[:1000]
    
    # Load job description
    with open('D:/resume-screener-mvp/data/job_descriptions/data_scientist.txt', 'r') as f:
        job_text = f.read()
    
    # Match and rank resumes
    start_time = time.time()
    ranked_results = match_resume_to_job(resume_texts, resume_filenames, job_text)
    print(f'Total processing time: {time.time() - start_time:.2f} seconds')
    
    # Save ranked results
    with open('D:/resume-screener-mvp/data/processed/ranked_resumes.json', 'w') as f:
        json.dump(ranked_results, f, indent=2)
    
    # Verify explanations for 100 resumes
    print('\nTop 5 resumes and their matched keywords:')
    for result in ranked_results[:5]:
        print(f"Resume: {result['filename']}, Similarity: {result['similarity']:.4f}, Keywords: {result['matched_keywords']}")
