import json
import re
import time
import logging
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResumeMatcher:
    """
    Resume-job matcher using sentence-transformers for semantic similarity and exact keyword matching.
    """
    
    def __init__(self, model_name: str = 'all-mpnet-base-v2', batch_size: int = 8, similarity_threshold: float = 0.2):
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.similarity_threshold = similarity_threshold
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english')).union({'employee', 'resume', 'cv', 'job', 'work', 'company', 'team'})
        self.job_specific_keywords = {
            'healthcare': {'patient care', 'vital signs', 'medical terminology', 'nursing', 'cpr certification', 'infection control', 'clinical skills', 'patient safety', 'healthcare compliance', 'electronic health records', 'patient advocacy'},
            'hr': {'recruitment', 'employee relations', 'benefits administration', 'labor law', 'performance management', 'hr policies', 'onboarding', 'talent acquisition', 'payroll management', 'hris', 'workplace diversity', 'conflict resolution', 'employee engagement', 'talent management', 'human resources', 'staffing'},
            'information-technology': {'network troubleshooting', 'cybersecurity', 'software installation', 'system diagnostics', 'windows', 'linux', 'cloud computing', 'it support'},
            'public-relations': {'media relations', 'content creation', 'event planning', 'crisis communication', 'public speaking', 'brand management', 'press releases'},
            'sales': {'sales techniques', 'crm software', 'customer relationship', 'negotiation', 'lead generation', 'sales forecasting', 'client acquisition'},
            'teacher': {'classroom management', 'lesson planning', 'student assessment', 'curriculum development', 'teaching', 'educational technology', 'student engagement'},
            'business-development': {'lead growth', 'relationship management', 'strategic planning', 'market analysis', 'business strategy', 'client partnerships', 'revenue growth'},
            'digital-media': {'social media', 'content creation', 'seo', 'google analytics', 'graphic design', 'digital marketing', 'content strategy', 'social media advertising'}
        }
        logger.info(f"Initialized ResumeMatcher with model '{model_name}', threshold {similarity_threshold}")

    def preprocess_text(self, text: str) -> str:
        tokens = word_tokenize(text.lower())
        bigrams = [' '.join(gram) for gram in ngrams(tokens, 2) if all(token.isalnum() or token in ['-', '.'] for token in gram)]
        trigrams = [' '.join(gram) for gram in ngrams(tokens, 3) if all(token.isalnum() or token in ['-', '.'] for token in gram)]
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token and (token.isalnum() or token in ['-', '.']) and token not in self.stop_words]
        processed_text = ' '.join(tokens + bigrams + trigrams)
        logger.debug(f"Preprocessed text sample: {processed_text[:200]}...")
        return processed_text

    def expand_job_description(self, job_text: str, job_name: str) -> str:
        expanded_text = job_text
        if job_name != "custom":
            job_keywords = self.job_specific_keywords.get(job_name, set())
            for keyword in job_keywords:
                expanded_text += f" {keyword}"
        context_additions = {
            'healthcare': " Coordinate patient scheduling, maintain medical records, adhere to healthcare regulations, implement infection control protocols, collaborate with multidisciplinary medical teams, ensure patient safety, utilize electronic health records systems, and provide patient-centered care.",
            'hr': " Manage HRIS systems, conduct employee training, ensure compliance with labor regulations, develop talent retention strategies, oversee performance reviews, facilitate workplace diversity initiatives, coordinate employee wellness programs, manage organizational culture, implement recruitment strategies, and handle employee onboarding processes.",
            'information-technology': " Diagnose and resolve network issues, implement cybersecurity measures, install and configure software, perform system diagnostics, manage Windows and Linux environments, and support cloud computing infrastructure.",
            'public-relations': " Develop media relations strategies, create engaging content, plan and execute events, manage crisis communication, deliver public speaking engagements, oversee brand management, and distribute press releases.",
            'sales': " Apply advanced sales techniques, utilize CRM software, build customer relationships, negotiate deals, generate leads, forecast sales, and acquire new clients.",
            'teacher': " Implement classroom management techniques, design lesson plans, assess student assessment, develop curricula, integrate educational technology, and enhance student engagement.",
            'business-development': " Drive lead growth, manage key relationships, develop strategic plans, conduct market analysis, formulate business strategies, foster client partnerships, and boost revenue growth.",
            'digital-media': " Manage social media campaigns, produce creative content, optimize SEO, analyze data with Google Analytics, design graphics, execute digital marketing strategies, and develop content plans."
        }
        expanded_text += context_additions.get(job_name, '')
        logger.info(f"Expanded job description for {job_name}: {expanded_text[:200]}...")
        return expanded_text.strip()

    def extract_keywords(self, text: str, job_text: str, job_name: str, max_keywords: int = 5) -> List[str]:
        """
        Extract keywords using exact matching from both job text and predefined keywords.
        """
        tokens = word_tokenize(text.lower())
        job_keywords = self.job_specific_keywords.get(job_name, set()) if job_name != "custom" else set()
        
        # Extract keywords from manual description if custom
        if job_name == "custom":
            job_tokens = word_tokenize(job_text.lower())
            job_keywords.update(job_token for job_token in job_tokens if job_token not in self.stop_words and len(job_token) > 2)
        
        matched_keywords = []
        for token in tokens:
            for job_kw in job_keywords:
                if token in job_kw or job_kw in token:
                    matched_keywords.append(job_kw)
        
        if not matched_keywords:
            logger.warning(f"No specific keywords matched for {job_name} in text. Falling back to TF-IDF.")
            vectorizer = TfidfVectorizer(max_features=100, stop_words=list(self.stop_words), ngram_range=(1, 3))
            tfidf_matrix = vectorizer.fit_transform([text, job_text])
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            matched_keywords = [feature_names[i] for i in tfidf_scores.argsort()[::-1][:max_keywords] if any(kw in feature_names[i] for kw in job_keywords) or feature_names[i] in job_text]
        
        return list(set(matched_keywords))[:max_keywords]

    def match_resume_to_job(self, resume_texts: List[str], resume_filenames: List[str], 
                            job_text: str, job_name: str = 'hr', max_keywords: int = 5) -> List[Dict]:
        if not resume_texts or not job_text:
            logger.warning("Empty input texts")
            return []

        job_text = self.expand_job_description(job_text, job_name)
        
        preprocessed_resumes = [self.preprocess_text(rt) for rt in resume_texts]
        preprocessed_job = self.preprocess_text(job_text)

        start_time = time.time()
        resume_embeddings = self.model.encode(preprocessed_resumes, batch_size=self.batch_size, show_progress_bar=True)
        job_embedding = self.model.encode([preprocessed_job])[0]
        embedding_time = time.time() - start_time
        logger.info(f"Embedding time: {embedding_time:.2f} seconds")

        similarities = util.cos_sim(job_embedding, resume_embeddings)[0]
        job_keywords = self.extract_keywords(job_text, job_text, job_name, max_keywords * 2)

        results = []
        for i, (filename, resume_text, similarity) in enumerate(zip(resume_filenames, resume_texts, similarities)):
            if similarity > self.similarity_threshold:
                resume_keywords = self.extract_keywords(resume_text, job_text, job_name, max_keywords)
                matched_keywords = [kw for kw in resume_keywords if kw in job_keywords or kw in job_text.lower().split()]
                explanation = f"Top matches: {', '.join(matched_keywords) if matched_keywords else 'general alignment'}. Overall semantic alignment: {similarity:.4f}."
                results.append({
                    'filename': filename,
                    'similarity': float(similarity),
                    'matched_keywords': matched_keywords,
                    'explanation': explanation
                })
                if not matched_keywords:
                    logger.debug(f"No specific keywords for {filename} (similarity: {similarity:.4f})")
            else:
                logger.debug(f"Resume {filename} below threshold ({similarity:.4f})")

        results = sorted(results, key=lambda x: x['similarity'], reverse=True)
        logger.info(f"Processed {len(results)} matches above threshold for {job_name}")
        return results

    def rank_multiple_jobs(self, resume_texts: List[str], resume_filenames: List[str], 
                           job_descriptions: Dict[str, str], max_results: int = 10) -> Dict:
        all_results = {}
        for job_name, job_text in job_descriptions.items():
            logger.info(f"Processing job: {job_name}")
            matches = self.match_resume_to_job(resume_texts, resume_filenames, job_text, job_name)
            all_results[job_name] = matches[:max_results]
        return all_results

if __name__ == '__main__':
    with open('D:/resume-screener-mvp/data/processed/extracted_texts.json', 'r') as f:
        resumes = json.load(f)
    
    resume_filenames = list(resumes.keys())[:100]  # Limit to 100 for testing
    resume_texts = list(resumes.values())[:100]   # Limit to 100 for testing
    
    with open('D:/resume-screener-mvp/data/job_descriptions/hr.txt', 'r') as f:
        job_text = f.read()
    
    matcher = ResumeMatcher(similarity_threshold=0.2, batch_size=8)
    start_time = time.time()
    ranked_results = matcher.match_resume_to_job(resume_texts, resume_filenames, job_text, job_name='hr')
    total_time = time.time() - start_time
    
    print(f"Processed {len(ranked_results)} matches in {total_time:.2f} seconds")
    print(f"Top 5 matches:")
    for result in ranked_results[:5]:
        print(f"Resume: {result['filename']}")
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"Explanation: {result['explanation']}")
        print("-" * 50)
    
    with open('D:/resume-screener-mvp/data/processed/advanced_matches_hr.json', 'w') as f:
        json.dump(ranked_results, f, indent=2)