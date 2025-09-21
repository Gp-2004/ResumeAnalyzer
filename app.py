import streamlit as st
import pdfplumber
import docx2txt
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from rapidfuzz import fuzz
import pandas as pd
from io import BytesIO

# ------------------------
# File Parsing Functions
# ------------------------
def extract_text(file):
    try:
        if file.type == "application/pdf":
            text = ""
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text.strip()
        elif "wordprocessingml" in file.type or "msword" in file.type:
            return docx2txt.process(file).strip()
        else:
            return ""
    except Exception as e:
        st.warning(f"Error reading file {file.name}: {e}")
        return ""

# ------------------------
# Resume Parsing Functions
# ------------------------
def extract_email(text):
    match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    return match.group(0) if match else "Not found"

def extract_phone(text):
    match = re.search(r'\+?\d[\d\s-]{7,}\d', text)
    return match.group(0) if match else "Not found"

def extract_skills(text, skills_list):
    found = []
    for skill in skills_list:
        if fuzz.token_set_ratio(skill.lower(), text.lower()) >= 75:
            found.append(skill)
    return found

def extract_name(text):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return lines[0] if lines else "Unknown"

# ------------------------
# Scoring Functions
# ------------------------
def compute_hard_score(must_skills, good_skills, resume_text):
    must_matches = len(extract_skills(resume_text, must_skills))
    good_matches = len(extract_skills(resume_text, good_skills))
    must_score = must_matches / max(1, len(must_skills))
    good_score = good_matches / max(1, len(good_skills))
    return 0.8 * must_score + 0.2 * good_score

# Load model once
@st.cache_resource(show_spinner=True)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

def embed_text(texts):
    try:
        return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    except Exception as e:
        st.warning(f"Embedding error: {e}")
        return np.zeros((len(texts), 384))  # fallback zero vector

def cosine_sim(a, b):
    eps = 1e-10
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + eps)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + eps)
    return np.dot(a_norm, b_norm.T)

def compute_semantic_score(jd_text, resume_text):
    resume_chunks = [p for p in resume_text.split("\n\n") if p.strip()][:50]
    texts = [jd_text] + resume_chunks
    embs = embed_text(texts)
    jd_emb = embs[0]
    resume_embs = embs[1:]
    sims = cosine_sim(jd_emb, resume_embs)
    top_sim = float(np.max(sims))
    avg_sim = float(np.mean(sims))
    return 0.7 * top_sim + 0.3 * avg_sim

def final_score(hard_score, semantic_score, hard_weight=0.45, semantic_weight=0.55):
    return hard_weight * hard_score * 100 + semantic_weight * semantic_score * 100

def verdict_from_score(score):
    if score >= 75:
        return "High ✅"
    elif score >= 50:
        return "Medium ⚠️"
    else:
        return "Low ❌"

# ------------------------
# Streamlit UI
# ------------------------
st.title("🚀 Batch Resume Evaluation System")
st.write("Upload multiple resumes (PDF/DOCX) and one Job Description.")

resume_files = st.file_uploader("Upload Resumes", type=["pdf", "docx"], accept_multiple_files=True)
jd_file = st.file_uploader("Upload Job Description", type=["pdf", "docx"])

jd_must_skills = st.text_input("Must-have skills (comma separated)").split(",")
jd_good_skills = st.text_input("Good-to-have skills (comma separated)").split(",")

if st.button("Evaluate Resumes"):
    if resume_files and jd_file:

        jd_text = extract_text(jd_file)
        if not jd_text:
            st.error("Job description could not be read. Please upload a valid file.")
        else:
            results = []

            for file in resume_files:
                resume_text = extract_text(file)
                if not resume_text:
                    st.warning(f"Could not read resume: {file.name}")
                    continue

                name = extract_name(resume_text)
                email = extract_email(resume_text)
                phone = extract_phone(resume_text)
                skills_found = extract_skills(resume_text, jd_must_skills + jd_good_skills)
                missing_skills = [s for s in jd_must_skills if s not in skills_found and s.strip()]

                # Safe scoring
                try:
                    hard = compute_hard_score(jd_must_skills, jd_good_skills, resume_text)
                except:
                    hard = 0
                try:
                    sem = compute_semantic_score(jd_text, resume_text)
                except:
                    sem = 0

                score = final_score(hard, sem)
                verdict = verdict_from_score(score)

                results.append({
                    "Name": name,
                    "Email": email,
                    "Phone": phone,
                    "Score": round(score,2),
                    "Verdict": verdict,
                    "Missing Skills": ", ".join(missing_skills)
                })

            if results:
                df = pd.DataFrame(results).sort_values(by="Score", ascending=False)
                st.subheader("📊 Evaluation Results")
                st.dataframe(df)

                # Export to Excel
                towrite = BytesIO()
                df.to_excel(towrite, index=False, sheet_name="Results")
                towrite.seek(0)
                st.download_button(
                    label="📥 Download Results as Excel",
                    data=towrite,
                    file_name="evaluation_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("No valid resumes to evaluate.")
    else:
        st.error("Please upload both resumes and job description.")
