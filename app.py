import streamlit as st
from utils import extract_text, summarize_text, build_qa_chain
import time

st.set_page_config(page_title="DocuSense AI", layout="wide")
st.title("📁 DocuSense – AI File Summarizer & QA")

uploadedFile = st.file_uploader("Upload a document", type=["pdf", "txt", "docx"])

def update_progress(statusText, progress):
    with st.status(statusText, expanded=True) as status:
        bar = st.progress(0)
        for i in range(1, progress + 1, 5):
            bar.progress(i)
            time.sleep(0.02)
        status.update(label=f"{statusText} Complete ✅", state="complete")

if uploadedFile:
    update_progress("📄 Extracting text...", 100)
    text = extract_text(uploadedFile, uploadedFile.name)

    st.subheader("📄 Extracted Text")
    st.text_area("Text", text, height=300)

    with st.expander("🧠 Summary"):
        update_progress("Summarizing...", 100)
        summary = summarize_text(text)
        st.success(summary)

    st.subheader("❓ Ask a Question")
    question = st.text_input("Ask anything from the document")

    if question:
        update_progress("Generating answer...", 100)
        answer = build_qa_chain(text, question)
        st.info(answer)
