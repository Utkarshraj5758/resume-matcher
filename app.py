import streamlit as st
import os
import tempfile
from matcher import match_resumes, get_job_descriptions
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# === Streamlit UI Setup ===
st.set_page_config(page_title="Resume Matcher", layout="wide")
st.title("📄 Resume Matcher with Groq LLM Insights")

# === Step 1: Choose or Enter Job Description ===
st.markdown("### 🧾 Job Description Input")

jd_input_method = st.radio(
    "How would you like to provide the job description?",
    ("Select from predefined roles", "Write your own description")
)

if jd_input_method == "Select from predefined roles":
    job_descriptions = get_job_descriptions()
    jd_title = st.selectbox("Select a Job Role", list(job_descriptions.keys()))
    jd_text = job_descriptions[jd_title]
else:
    jd_title = "Custom Description"
    jd_text = st.text_area("Paste or type your custom Job Description here", height=200)

# === Step 2: Upload Resumes ===
uploaded_files = st.file_uploader(
    "Upload one or more resumes (.pdf or .docx)",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

# === Step 3: Matching Parameters ===
st.markdown("### ⚙️ Matching Parameters")
col1, col2 = st.columns(2)

with col1:
    cosine_threshold = st.slider(
        "Cosine Similarity Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.75, 
        step=0.01,
        help="Only resumes with similarity above this threshold will be evaluated by the LLM."
    )

with col2:
    top_k = st.slider(
        "Number of Top Matches", 
        min_value=1, 
        max_value=10, 
        value=5, 
        step=1,
        help="Maximum number of top matching resumes to show."
    )

# === Step 4: Match Button ===
if st.button("🔍 Find Matching Resumes"):
    if not uploaded_files:
        st.error("❌ Please upload at least one resume.")
    elif not jd_text.strip():
        st.error("❌ Please provide a job description.")
    else:
        with st.spinner("Matching resumes and generating insights..."):
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded resumes to temporary directory
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.read())

                # Run matching
                top_matches = match_resumes(
                    jd_text,
                    resume_folder=temp_dir,
                    use_groq=True,
                    top_k=top_k,
                    cosine_threshold=cosine_threshold
                )

        # === Step 5: Display Results ===
        if not top_matches:
            st.warning("⚠️ No matching resumes passed the similarity threshold.")
        else:
            st.success(f"✅ Top {len(top_matches)} Matches for **{jd_title}**")
            for file_name, final_score, insight, cosine_score, fit_score in top_matches:
                with st.expander(f"📄 {file_name} — Final Score: {final_score:.4f}"):
                    st.markdown(f"**🔹 Cosine Similarity:** {cosine_score:.4f}")
                    st.markdown(
                        f"**🤖 LLM Fit Score:** {fit_score:.1f} / 10" 
                        if fit_score is not None else "**🤖 LLM Fit Score:** N/A"
                    )
                    st.markdown("### 💡 AI Insights")
                    st.markdown(insight or "_No insights available._")