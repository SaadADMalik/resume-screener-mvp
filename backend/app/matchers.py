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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download resources
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

class ResumeRanker:
    def __init__(self, similarity_threshold: float = 0.2, batch_size: int = 8):
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        self.model = SentenceTransformer("all-mpnet-base-v2")
        self.lemmatizer = WordNetLemmatizer()

        # Keep stopwords but don’t kill role words
        self.stop_words = set(stopwords.words("english")).union({
            "hr", "employee", "skill", "experience", "resume", "cv",
            "job", "work", "professional", "company", "team", "and"
        })

        # Fallback synonyms
        self.synonyms = {
            "python": ["py"],
            "sql": ["structured query language"],
            "machine learning": ["ml", "ai"],
            "developer": ["programmer", "coder", "engineer"],
            "manager": ["lead", "supervisor"]
        }

        # Regex patterns for impact phrases
        self.impact_verbs = re.compile(
            r"\b(managed|led|built|increased|reduced|created|designed|taught|organized|developed|closed|negotiated|delivered|launched)\b",
            re.I,
        )
        self.impact_numbers = re.compile(
            r"(\d+%|\$?\d+(?:,\d+)*(?:\.\d+)?\s*(million|billion|k)?|\d+\syears?)",
            re.I,
        )

    # ---------- UTILITIES ----------
    def preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        tokens = [self.lemmatizer.lemmatize(w) for w in text.split() if w not in self.stop_words]
        return " ".join(tokens)

    def extract_skills_section(self, text: str) -> str:
        """Extract skills or bullet-like sections"""
        lines = text.splitlines()
        skills_text = []
        skill_indicators = re.compile(
            r"\b(skills|competencies|abilities|expertise|proficiencies|technical skills|key skills|skill set|core competencies)\b",
            re.I,
        )

        for i, line in enumerate(lines):
            if skill_indicators.search(line):
                if ":" in line:
                    after = line.split(":", 1)[1].strip()
                    if after:
                        skills_text.append(after)
                for j in range(i + 1, min(len(lines), i + 6)):
                    if lines[j].strip():
                        skills_text.append(lines[j].strip())
                    else:
                        break
        if skills_text:
            return " ".join(skills_text)

        # fallback: detect bullets / comma-separated skills
        candidates = [
            ln.strip()
            for ln in lines
            if re.search(r"[\•\-\u2022,]", ln) and len(ln.split()) < 60
        ]
        if candidates:
            return " ".join(candidates[:6])
        return text

    def extract_keywords(self, job_text: str, job_name: str, max_keywords: int = 15) -> List[str]:
        """Extract top keywords from job description dynamically"""
        preprocessed = self.preprocess_text(job_text + " " + job_name)
        documents = [preprocessed]
        vectorizer = TfidfVectorizer(max_features=50)
        tfidf_matrix = vectorizer.fit_transform(documents)
        keywords = vectorizer.get_feature_names_out()
        return list(keywords)[:max_keywords]

    def detect_impact_score(self, text: str) -> float:
        """Boost for resumes that show measurable impact"""
        score = 0
        if self.impact_verbs.search(text):
            score += 0.5
        if self.impact_numbers.search(text):
            score += 0.5
        return min(score, 1.0)  # cap at 1.0

    # ---------- RANKING ----------
    def rank_resumes(
        self,
        resume_texts: List[str],
        resume_filenames: List[str],
        job_text: str,
        job_name: str,
        max_keywords: int = 15,
    ) -> List[Dict]:
        if not resume_texts:
            return []

        # Encode job description
        job_embedding = self.model.encode(job_text + " " + job_name, convert_to_tensor=True)
        resume_embeddings = self.model.encode(
            resume_texts, convert_to_tensor=True, batch_size=self.batch_size
        )
        similarities = util.cos_sim(job_embedding, resume_embeddings)[0].tolist()

        # Extract dynamic keywords from JD
        job_keywords = self.extract_keywords(job_text, job_name, max_keywords)

        results = []
        alpha, beta, gamma = 0.6, 0.3, 0.1  # semantic, keyword, impact weights

        for filename, resume_text, similarity in zip(resume_filenames, resume_texts, similarities):
            resume_lower = resume_text.lower()

            # Keyword overlap
            matched_keywords = []
            for kw in job_keywords:
                if kw in resume_lower or fuzz.partial_ratio(kw, resume_lower) > 70:
                    matched_keywords.append(kw)
            kw_score = len(matched_keywords) / (len(job_keywords) if job_keywords else 1)

            # Impact score
            impact_score = self.detect_impact_score(resume_text)

            # Weighted score
            combined_score = (alpha * float(similarity)) + (beta * kw_score) + (gamma * impact_score)

            if combined_score >= max(self.similarity_threshold, 0.15):
                results.append({
                    "filename": filename,
                    "similarity": float(similarity),
                    "kw_score": kw_score,
                    "impact_score": impact_score,
                    "combined_score": combined_score,
                    "matched_keywords": matched_keywords[:5],  # top few matches
                    "explanation": (
                        f"Semantic={similarity:.3f}, KW={kw_score:.2f}, "
                        f"Impact={impact_score:.2f}, Combined={combined_score:.3f}"
                    ),
                })

        results.sort(key=lambda x: x["combined_score"], reverse=True)
        return results[:10]
