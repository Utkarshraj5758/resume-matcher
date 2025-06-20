import os
import re
import fitz  # PyMuPDF for accurate PDF extraction
import docx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from dotenv import load_dotenv
from streamlit import secrets
import numpy as np

# Load environment variables
load_dotenv()

# === Load model and Groq client ===
model = SentenceTransformer("all-MiniLM-L6-v2")
groq_client = Groq(api_key=secrets["GROQ_API_KEY"])

# === Job Descriptions Function ===
def get_job_descriptions():
    return {
        "Python Developer": "Seeking a Python developer with experience in Flask, REST APIs, and SQL. Knowledge of cloud platforms like AWS or Azure is a plus.",
        "Data Analyst": "Looking for a data analyst skilled in Python, SQL, Excel, and Tableau. Must understand statistics and reporting.",
        "DevOps Engineer": "Need a DevOps engineer with experience in Docker, Kubernetes, CI/CD, Jenkins, and cloud infrastructure.",
        "Frontend Developer": "Require a frontend developer experienced in React, JavaScript, HTML, CSS, and UI/UX best practices.",
        "Machine Learning Engineer": "Hiring a machine learning engineer skilled in Python, scikit-learn, TensorFlow or PyTorch. Knowledge of data pipelines and model evaluation needed.",
        "Cybersecurity Analyst": "Searching for a cybersecurity analyst with experience in threat analysis, risk mitigation, network security, and SIEM tools."
    }

# === Text Preprocessing ===
def clean_text(text):
    text = re.sub(r"\s+", " ", text)  # normalize spaces
    text = re.sub(r"[^a-zA-Z0-9\s.,;:?!@#\-_/]", "", text)  # keep useful symbols
    return text.strip().lower()

def emphasize_skills(text, skills):
    return text + "\n\nSkills:\n" + " ".join(skills)

def extract_skills_from_jd(jd):
    return re.findall(r"\b[A-Za-z]+\b", jd)

# === Resume Text Extraction ===
def extract_text_from_pdf(file_path):
    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
    except Exception as e:
        print(f"[PDF ERROR] {file_path}: {e}")
    return text

def extract_text_from_docx(file_path):
    try:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"[DOCX ERROR] {file_path}: {e}")
        return ""

def extract_resume_text(file_path):
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    return ""

# === Groq Insight Generator ===
def get_resume_insights_from_groq(jd, resume_text):
    prompt = f"""
    Based on the following Job Description and Resume, provide:
    - 3 strengths of the candidate
    - 3 weaknesses
    - A 1-line summary of the candidate's fit for this role

    Job Description:
    {jd}

    Resume:
    {resume_text}
    """
    try:
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are an expert recruitment assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Groq Insight Error]: {e}")
        return None

# === Groq Fit Score ===
def get_fit_score_from_groq(jd, resume_text):
    prompt = f"""
    Given the following Job Description and Resume, rate the candidate's overall fit for the role on a scale of 1 to 10.
    Respond only with the number.

    Job Description:
    {jd}

    Resume:
    {resume_text}
    """
    try:
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are an expert recruiter."},
                {"role": "user", "content": prompt}
            ]
        )
        score_str = response.choices[0].message.content.strip()
        return float(score_str)
    except Exception as e:
        print(f"[Groq Score Error]: {e}")
        return None

# === Resume Matching Function ===
import concurrent.futures

def match_resumes(jd_text, resume_folder, use_groq=True, top_k=5, cosine_threshold=0.75):
    jd_text_clean = clean_text(jd_text)
    skills = extract_skills_from_jd(jd_text_clean)
    jd_embed = model.encode(jd_text_clean)
    matches = []

    resume_files = [
        os.path.join(resume_folder, file)
        for file in os.listdir(resume_folder)
        if file.lower().endswith((".pdf", ".docx"))
    ]

    texts = [extract_resume_text(path) for path in resume_files]
    valid_files = [(file, text) for file, text in zip(resume_files, texts) if text.strip()]
    if not valid_files:
        print("[ERROR] No valid resume text extracted.")
        return []

    processed_texts = [
        clean_text(emphasize_skills(text, skills)) for _, text in valid_files
    ]
    embeddings = model.encode(processed_texts)

    cosine_results = []
    for i, (file_path, original_text) in enumerate(valid_files):
        file_name = os.path.basename(file_path)
        cosine_score = cosine_similarity([jd_embed], [embeddings[i]])[0][0]
        print(f"{file_name} — Cosine Similarity: {cosine_score:.4f}")
        if cosine_score >= cosine_threshold:
            cosine_results.append((file_name, file_path, original_text, cosine_score))
        else:
            print(f"🔻 Skipping {file_name} due to low cosine similarity")

    if not cosine_results:
        print("[INFO] No resumes passed cosine threshold.")
        return []

    # Sort and select top K
    cosine_results.sort(key=lambda x: x[3], reverse=True)
    selected = cosine_results[:top_k]

    # === Run Groq calls in parallel ===
    def process_resume_groq(file_name, resume_text, cosine_score):
        fit_score = get_fit_score_from_groq(jd_text, resume_text) if use_groq else None
        insight = get_resume_insights_from_groq(jd_text, resume_text) if use_groq else None
        final_score = (cosine_score * 0.6) + ((fit_score / 10) * 0.4) if fit_score is not None else cosine_score
        return (file_name, final_score, insight, cosine_score, fit_score)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(process_resume_groq, file_name, resume_text, cosine_score)
            for file_name, _, resume_text, cosine_score in selected
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                matches.append(result)
            except Exception as e:
                print(f"[Parallel Error]: {e}")

    # Sort final matches by score
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches