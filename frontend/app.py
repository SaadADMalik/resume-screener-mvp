import streamlit as st
import requests
import base64
import os
import pdfplumber

st.set_page_config(page_title="📊 Resume Ranker", layout="wide")
st.title("📊 Resume Ranker")

# Backend URL
backend_url = "http://localhost:8000/rank-resumes"

# --- Load saved job descriptions ---
job_desc_dir = os.path.join("data", "job_descriptions")
saved_jobs = {}
if os.path.exists(job_desc_dir):
    for fname in os.listdir(job_desc_dir):
        if fname.endswith(".txt"):
            with open(os.path.join(job_desc_dir, fname), "r", encoding="utf-8") as f:
                saved_jobs[fname.replace(".txt", "")] = f.read()

mode = st.radio("Choose mode:", ["Use saved job description", "Paste job description"])

job_text, job_name = "", ""
if mode == "Use saved job description":
    if not saved_jobs:
        st.warning("No job descriptions found in data/job_descriptions/")
    else:
        job_name = st.selectbox("Select Job Title", list(saved_jobs.keys()))
        job_text = saved_jobs[job_name]
        st.text_area("Job Description (auto-loaded)", value=job_text, height=300)
else:
    job_name = st.text_input("Job Title / Category")
    job_text = st.text_area("Paste Job Description", height=300)

# --- Upload resumes ---
uploaded_files = st.file_uploader("Upload CVs (PDF)", type="pdf", accept_multiple_files=True)

if st.button("Rank Resumes"):
    if not uploaded_files or not job_text.strip() or not job_name.strip():
        st.error("Please upload resumes and provide job details.")
    else:
        resume_texts, resume_filenames, resume_files_b64 = [], [], []
        for uploaded_file in uploaded_files:
            resume_filenames.append(uploaded_file.name)

            # Try quick text extraction
            text = ""
            try:
                with pdfplumber.open(uploaded_file) as pdf:
                    for page in pdf.pages:
                        text += page.extract_text() or ""
            except Exception:
                text = ""

            resume_texts.append(text)
            uploaded_file.seek(0)
            resume_files_b64.append(base64.b64encode(uploaded_file.read()).decode("utf-8"))

            st.write(f"{uploaded_file.name}: extracted text length = {len(text)}")

        payload = {
            "resume_texts": resume_texts,
            "resume_filenames": resume_filenames,
            "resume_files_b64": resume_files_b64,
            "job_text": job_text,
            "job_name": job_name,
            "max_results": 5,
        }

        with st.spinner("Ranking resumes..."):
            try:
                response = requests.post(backend_url, json=payload)
                response.raise_for_status()
                results = response.json()["results"]
                if results:
                    st.subheader("Results")
                    for res in results:
                        st.markdown(f"### {res['filename']}")
                        st.write(res["explanation"])
                        if res.get("matched_keywords"):
                            st.write("**Matched Keywords:**", ", ".join(res["matched_keywords"]))
                else:
                    st.warning("No relevant matches found.")
            except Exception as e:
                st.error(f"Error: {e}")
