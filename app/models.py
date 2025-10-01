import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ResumeModel:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)

    def compute_similarity(self, resume_embeddings, job_embedding):
        return cosine_similarity(resume_embeddings, [job_embedding])

if __name__ == '__main__':
    # Load extracted resume texts
    with open('D:/resume-screener-mvp/data/processed/extracted_texts.json', 'r') as f:
        resumes = json.load(f)
    
    # Select 10 resumes
    resume_texts = list(resumes.values())[:10]
    resume_filenames = list(resumes.keys())[:10]
    
    # Load job description
    with open('D:/resume-screener-mvp/data/job_descriptions/data_scientist.txt', 'r') as f:
        job_text = f.read()
    
    # Initialize model
    model = ResumeModel()
    
    # Embed texts
    resume_embeddings = model.embed_texts(resume_texts)
    job_embedding = model.embed_texts([job_text])[0]
    
    # Compute similarities
    similarities = model.compute_similarity(resume_embeddings, job_embedding)
    
    # Print results
    for filename, similarity in zip(resume_filenames, similarities):
        print(f'Resume: {filename}, Similarity: {similarity[0]:.4f}')
