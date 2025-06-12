import fitz  # PyMuPDF
from transformers import pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def extract_text(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def summarize_text(text):
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    return summarizer(text[:1024])[0]["summary_text"]

def build_qa_chain(text, question):
    texts = CharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text(text)
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_texts(texts, embed_model)
    retriever = vectordb.as_retriever()
    llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0})
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa.run(question)
