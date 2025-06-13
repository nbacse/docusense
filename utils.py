import os
import fitz  # PyMuPDF
import docx
from typing import List
from transformers import pipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
import streamlit as st

if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

def _extract_pdf(file) -> str:
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

def _extract_docx(file) -> str:
    document = docx.Document(file)
    return "\n".join(p.text for p in document.paragraphs)

def _extract_txt(file) -> str:
    return file.read().decode("utf-8", errors="ignore")

def extract_text(file, filename: str) -> str:
    ext = filename.split(".")[-1].lower()
    if ext == "pdf":
        return _extract_pdf(file)
    elif ext == "docx":
        return _extract_docx(file)
    elif ext == "txt":
        return _extract_txt(file)
    else:
        raise ValueError("Unsupported file type")

@st.cache_resource(show_spinner=False)
def _load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize_text(text: str, max_chars: int = 1024) -> str:
    summarizer = _load_summarizer()
    if len(text) <= max_chars:
        return summarizer(text)[0]["summary_text"]
    chunks = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
    summaries = [summarizer(c)[0]["summary_text"] for c in chunks]
    joined = " ".join(summaries)
    return summarizer(joined)[0]["summary_text"] if len(joined) > max_chars else joined

@st.cache_resource(show_spinner=False)
def _load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def _load_llm():
    return HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0})

def build_qa_chain(text: str, question: str) -> str:
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_text(text)
    store = FAISS.from_texts(docs, _load_embeddings())
    qa = RetrievalQA.from_chain_type(llm=_load_llm(), retriever=store.as_retriever())
    return qa.run(question)
