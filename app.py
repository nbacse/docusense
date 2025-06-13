import streamlit as st
import time
from utils import extract_text, summarize_text, build_qa_chain

st.set_page_config(page_title="DocuSense AI", layout="wide", page_icon="📁")
st.title("📁 DocuSense – AI File Summarizer & QA")

uploaded_file = st.file_uploader("Upload a document (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

def run_with_progress(label, func, *args, **kwargs):
    progress_placeholder = st.empty()
    progress_bar = progress_placeholder.progress(0, text=label)
    for pct in range(0, 100, 10):
        time.sleep(0.05)
        progress_bar.progress(pct + 1, text=label)
    result = func(*args, **kwargs)
    progress_bar.progress(100, text=f"{label} ✅")
    time.sleep(0.1)
    progress_placeholder.empty()
    return result

if uploaded_file:
    text = run_with_progress("Extracting text...", extract_text, uploaded_file, uploaded_file.name)
    st.subheader("📄 Extracted Text")
    st.text_area("Preview", text, height=300)
    summary = run_with_progress("Summarizing...", summarize_text, text)
    st.subheader("🧠 Summary")
    st.success(summary)
    st.subheader("❓ Ask a Question")
    question = st.text_input("Ask anything about this document")
    if question:
        answer = run_with_progress("Generating answer...", build_qa_chain, text, question)
        st.info(answer)
