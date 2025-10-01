# backend/app/matchers.py

import spacy
import re
import logging
from typing import List, Dict
import torch
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from io import BytesIO
import pdfplumber

# ---------- LOGGING ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- NLTK SETUP ----------
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# ---------- SPACY SETUP ----------
# Load the model from the copied location
try:
    nlp = spacy.load("en_core_web_sm")  # Load the model from the standard location (installed during build)
except OSError:
    logger.error("SpaCy model not found. Please ensure en_core_web_sm is downloaded.")
    raise

# ---------- CLASS DEFINITION ----------
class ResumeRanker:
    def __init__(self, similarity_threshold: float = 0.0, batch_size: int = 8):
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        self.model = SentenceTransformer("all-mpnet-base-v2")  # Restored
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

        # Regex patterns for impact scoring
        self.impact_verbs = re.compile(
            r"\b(managed|led|built|increased|reduced|created|designed|organized|developed|negotiated|delivered|launched)\b",
            re.I
        )
        self.impact_numbers = re.compile(
            r"(\d+%|\$?\d+(?:,\d+)*(?:\.\d+)?\s*(million|billion|k)?|\d+\syears?)",
            re.I
        )

    # ---------- Extract text from PDF ----------
    def extract_text_from_pdf(self, pdf_file: BytesIO) -> str:
        text = ""
        try:
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
        except Exception:
            pass
        return text

    # ---------- Text preprocessing ----------
    def preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        tokens = [
            self.lemmatizer.lemmatize(w)
            for w in text.split()
            if w not in self.stop_words
        ]
        return " ".join(tokens)

    # ---------- Keyword extraction ----------
    def extract_keywords(self, job_text: str, job_name: str, max_keywords: int = 20) -> List[str]:
        text = job_text + " " + job_name
        doc = nlp(text.lower())
        phrases = set()

        # Multi-word noun chunks
        for chunk in doc.noun_chunks:
            phrase = chunk.text.strip()
            if len(phrase.split()) > 1:
                phrases.add(phrase)

        # Single-word lemmatized tokens
        tokens = [t.lemma_ for t in doc if not t.is_stop and t.is_alpha and len(t.text) > 2]
        phrases.update(tokens)

        # TF-IDF top keywords
        vectorizer = TfidfVectorizer(max_features=50)
        vectorizer.fit([self.preprocess_text(text)])
        tfidf_keywords = vectorizer.get_feature_names_out()
        phrases.update(tfidf_keywords)

        # Sort: multi-word first
        phrases = sorted(phrases, key=lambda x: (-len(x.split()), x))
        return list(phrases)[:max_keywords]

    # ---------- Impact scoring ----------
    def detect_impact_score(self, resume_text: str) -> float:
        if not resume_text:
            return 0.0
        verbs = len(self.impact_verbs.findall(resume_text.lower()))
        numbers = len(self.impact_numbers.findall(resume_text))
        impact = (min(verbs, 5) / 5 + min(numbers, 5) / 5) / 2
        return min(impact, 0.5)

    # ---------- Main ranking ----------
    def rank_resumes(
        self,
        resume_texts: List[str],
        resume_filenames: List[str],
        job_text: str,
        job_name: str,
        max_keywords: int = 20,
        top_k: int = None
    ) -> List[Dict]:
        if not resume_texts:
            return []

        job_embedding = self.model.encode(job_text + " " + job_name, convert_to_tensor=True)
        resume_embeddings = self.model.encode(resume_texts, convert_to_tensor=True, batch_size=self.batch_size)
        similarities = util.cos_sim(job_embedding, resume_embeddings)[0].tolist()

        job_keywords = self.extract_keywords(job_text, job_name, max_keywords)
        results = []
        alpha, beta, gamma = 0.6, 0.3, 0.1

        for filename, resume_text, similarity in zip(resume_filenames, resume_texts, similarities):
            resume_lower = resume_text.lower()
            matched_keywords = []

            for kw in job_keywords:
                if kw in resume_lower:
                    matched_keywords.append(kw)

            matched_keywords = list(set(matched_keywords))[:10]

            kw_score = len(matched_keywords) / (len(job_keywords) if job_keywords else 1)
            impact_score = self.detect_impact_score(resume_text)
            combined_score = (alpha * float(similarity)) + (beta * kw_score) + (gamma * impact_score)

            if combined_score >= self.similarity_threshold:
                results.append({
                    "filename": filename,
                    "similarity": float(similarity),
                    "kw_score": kw_score,
                    "impact_score": impact_score,
                    "combined_score": combined_score,
                    "matched_keywords": matched_keywords,
                    "explanation": (
                        f"Semantic={similarity:.3f}, KW={kw_score:.2f}, "
                        f"Impact={impact_score:.2f}, Combined={combined_score:.3f}"
                    ),
                })

        results.sort(key=lambda x: x["combined_score"], reverse=True)
        return results[:top_k] if top_k else results