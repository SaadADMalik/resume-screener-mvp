📄Resume Screener

An end-to-end AI-powered application that ranks resumes against job descriptions using semantic similarity, keyword relevance, and quantified impact scoring. Built with a modular full-stack architecture (Streamlit + FastAPI) and fully containerized with Docker for reproducibility and deployment.

🚀 Features

Upload multiple resumes (PDFs) and match them against a pasted or uploaded job description.

Hybrid ranking engine:

🔹 Semantic similarity with SentenceTransformers
 (all-mpnet-base-v2).

🔹 Keyword relevance using TF-IDF, SpaCy lemmatization, and fuzzy matching.

🔹 Impact scoring that rewards quantified achievements (e.g., “reduced costs by 20%”).

Explainable results with matched keywords and scoring breakdowns.

Batch processing: Rank multiple resumes in a single API request.

Streamlit UI for recruiters, FastAPI backend for compute-heavy tasks.

Dockerized deployment for consistent environments and cloud hosting.

🏗️ Architecture (only useful files)
📂 resume-screener
├── frontend/          # Streamlit app for recruiter interaction
│   └── app.py
├── backend/           # FastAPI service
│   ├── main.py        # API endpoints
│   ├── ranking.py     # Scoring logic
│   ├── nlp_utils.py   # Embeddings, TF-IDF, SpaCy preprocessing
├── models/            # Pretrained SentenceTransformer + fine-tuning (optional)
├── docker/            # Dockerfiles and compose setup
└── README.md


Streamlit → File upload, JD input, results visualization.

FastAPI + Uvicorn → Stateless API with NLP pipelines.

Docker → Containerized environment for reproducibility and scalability.

⚙️ Tech Stack

Backend Framework: FastAPI (ASGI server with Uvicorn)

Frontend: Streamlit

NLP Models:

SentenceTransformers (all-mpnet-base-v2) for embeddings

SpaCy (en_core_web_sm) for tokenization + lemmatization

Scikit-learn (TF-IDF Vectorizer)

FuzzyWuzzy for approximate string matching

PDF Handling: PDFPlumber

Text Preprocessing: NLTK (stopwords, lemmatization)

Containerization: Docker

🔬 Ranking Methodology

The ranking function balances three dimensions:

Semantic Similarity (60%) → Contextual embedding similarity with SentenceTransformers.

Keyword Matching (30%) → TF-IDF + SpaCy noun-chunk matching + fuzzy string overlap.

Impact Scoring (10%) → Regex-based detection of action verbs + numerical achievements.

Example:
Resume A: “Improved sales by 25% using data-driven strategies.”
Resume B: “Responsible for sales team tasks.”
→ Resume A scores higher because of quantified, action-driven phrasing.

📊 Explainability

Recruiters receive scoring breakdowns with:

Matched keywords per resume

Explanation of semantic similarity scores

Highlighted impact phrases

This turns the model into a transparent AI assistant, not a black box.

🚢 Deployment

Local:

docker-compose up --build


Cloud-ready: Designed for free-tier hosting (Render, Fly.io, Railway).

Stateless backend → can scale horizontally with multiple Docker workers.

🧠 Future Enhancements

 Fine-tune SentenceTransformers on resume–JD pairs (domain adaptation).

 Add retrieval evaluation metrics (MRR, nDCG, Precision@k).

 Integrate with ATS platforms via API.

 Support multilingual resumes and job descriptions.

 Deploy on Kubernetes for production-scale workloads.

🏅 Why This Project Matters

This project goes beyond simple keyword filtering by combining semantic NLP, explainability, and containerized deployment. It’s designed as a production-ready AI tool, not just a prototype — balancing algorithmic sophistication with real-world usability.
