import streamlit as st
from utils import extract_text, summarize_text, build_qa_chain

st.set_page_config(page_title="DocuSense", layout="wide")

st.title("📄 DocuSense - AI Document Intelligence Assistant")

uploaded_file = st.file_uploader("Upload a PDF or Image", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    with st.spinner("Extracting text..."):
        text = extract_text(uploaded_file)

    st.subheader("📑 Extracted Text")
    st.text_area("", text, height=200)

    with st.spinner("Summarizing..."):
        summary = summarize_text(text)
    st.subheader("🔍 Summary")
    st.write(summary)

    st.subheader("🤖 Ask a Question")
    question = st.text_input("Ask something about this document")
    if question:
        with st.spinner("Thinking..."):
            answer = build_qa_chain(text, question)
        st.success(answer)
