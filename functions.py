import streamlit as st
import os
import datetime
import chromadb
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from dotenv import load_dotenv
import sqlite3 as s
import hashlib

# Carregar variáveis de ambiente
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configurar ChromaDB local
chroma_client = chromadb.PersistentClient(path="./chroma_db")
def vector_store():
    vectorstore = Chroma(
        collection_name="documents",  # Coleção única para armazenar múltiplos arquivos
        embedding_function=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
        persist_directory="./chroma_db"
    )
    return vectorstore

vectorstore = vector_store()

# Criar função para processar arquivos com metadados
def process_file(file):
    ext = file.name.split(".")[-1]
    temp_path = f"./temp_uploaded.{ext}"  # Caminho temporário
    
    # Salvar arquivo temporário
    with open(temp_path, "wb") as f:
        f.write(file.getbuffer())
    
    if ext == "pdf":
        loader = PyPDFLoader(temp_path)
    elif ext in ["doc", "docx"]:
        loader = UnstructuredWordDocumentLoader(temp_path)
    else:
        st.error("Formato de arquivo não suportado")
        return []
    
    documents = loader.load()
    os.remove(temp_path)  # Remover arquivo temporário após uso

    text = "\n".join([doc.page_content for doc in documents])
    file_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    existing_docs = vectorstore.get(
        where={"file_hash": file_hash},
        limit=1)

    if existing_docs["documents"]:
        return []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    # Adicionar metadados com base no nome do arquivo
    for doc in docs:
        doc.metadata.update({
            "file_hash": file_hash, 
            "source": file.name,
            "length": len(doc.page_content),
            "upload_date": str(datetime.datetime.now())
        })
    
    return docs

# Banco de dados para armazenar os feedbacks
def init_db():
    conn = s.connect("feedback.db")
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS feedbacks(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              question TEXT,
              answer TEXT,
              rating INTEGER,
              source TEXT
    )
    ''')
    conn.commit()
    conn.close()

def store_feedback(question, answer, rating, source):
    conn = s.connect("feedback.db")
    c = conn.cursor()
    c.execute('''
    INSERT INTO feedbacks (question, answer, rating, source)
    VALUES (?, ?, ?, ?)
    ''', (question, answer, rating, source))
    conn.commit()
    conn.close()