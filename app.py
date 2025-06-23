import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def create_vector_store(text, path="faiss_index"):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local(path)
    
    return vector_store

def load_vector_store(path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(path, 
                                    embeddings, 
                                    allow_dangerous_deserialization=True)
    return vector_store

def build_rag_chain(path="faiss_index"):
    vector_store = load_vector_store(path)
    retriever = vector_store.as_retriever()
    # Build the QA chain
    llm = Ollama(model="llama3.2:3B")
    qa_chain = load_qa_chain(llm, chain_type="stuff")
    rag = RetrievalQA(retriever=retriever, combine_documents_chain=qa_chain)
    return rag

st.title("AI RAG app with LangChain and Llama")

st.write("Upload a PDF file and ask questions about its content.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    pdf_path = f"uploaded/{uploaded_file.name}"
    os.makedirs("uploaded", exist_ok=True)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("File uploaded successfully!")

    text = extract_text_from_pdf(pdf_path)

    st.info("Creating vector store...")
    
    create_vector_store(text)

    st.info("Iniitializing AI RAG App...")

    rag = build_rag_chain()

    st.success("AI RAG App is ready!")

    if 'rag' in locals():
        question = st.text_input("Ask a question about the uploaded PDF:")
        
        if question:
            with st.spinner("Getting answer..."):
                answer = rag.run(question)
                st.success(f"Answer: {answer}")
