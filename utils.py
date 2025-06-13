import os
import fitz  # PyMuPDF
import docx
from transformers import pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
import streamlit as st

os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

def extract_text(file, filename):
    extension = filename.split('.')[-1].lower()

    if extension == "pdf":
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join(page.get_text() for page in doc)

    elif extension == "txt":
        return file.read().decode("utf-8")

    elif extension == "docx":
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])

    else:
        return "Unsupported file format."

def summarize_text(text):
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    return summarizer(text[:1024])[0]["summary_text"]

def build_qa_chain(text, question):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_texts(chunks, embeddings)
    retriever = vectordb.as_retriever()

    llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0})
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    return qa.run(question)
