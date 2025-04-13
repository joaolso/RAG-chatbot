import streamlit as st
import os
import chromadb
import datetime
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import sqlite3 as s
import re
import hashlib

# Carregar variáveis de ambiente
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configurar ChromaDB local
chroma_client = chromadb.PersistentClient(path="./chroma_db")
vectorstore = Chroma(
    collection_name="documents",  # Coleção única para armazenar múltiplos arquivos
    embedding_function=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
    persist_directory="./chroma_db"
)

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

init_db()

def store_feedback(question, answer, rating, source):
    conn = s.connect("feedback.db")
    c = conn.cursor()
    c.execute('''
    INSERT INTO feedbacks (question, answer, rating, source)
    VALUES (?, ?, ?, ?)
    ''', (question, answer, rating, source))
    conn.commit()
    conn.close()

# Parâmetros da LLM definidos estaticamente
LLM_PARAMS = {
    "temperature": 0.7,
    "max_tokens": 512,
    "top_p": 0.95,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
}

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

# Interface Streamlit
st.set_page_config(page_title="Chatbot RAG", layout="wide")
st.title("Chatbot RAG com Streamlit e ChromaDB")

# Sidebar para upload de múltiplos arquivos
with st.sidebar:
    st.header("Upload de Documentos")
    uploaded_files = st.file_uploader("Faça upload de PDFs ou Word", type=["pdf", "docx"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.write(f"Processando arquivo: {uploaded_file.name}...")
            docs = process_file(uploaded_file)
            if docs:
                vectorstore.add_documents(docs)
                st.success("Todos os arquivos foram processados e indexados com sucesso!")
            else:
                st.error("O arquivo já existe no banco")


# Histórico de chat usando st.chat_message
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def save_feedback(idx):
    rating = st.session_state[f"feedback_{idx}"]
    chat =  st.session_state.chat_history[idx]
    if auxiliar == 1:
        store_feedback(chat['pergunta'], chat['resposta'], int(rating), referencia)
    else:
        store_feedback(chat['pergunta'], chat['resposta'], int(rating), "")

    st.toast("Obrigado pelo feedback!", icon="✅")

st.subheader("Chat com os Documentos")
with st.chat_message("assistant"):
    st.markdown("Olá! Envie sua pergunta sobre os documentos carregados.")

auxiliar = 0
query = st.chat_input("Digite sua pergunta...")
if query:
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, **LLM_PARAMS)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Buscar os 3 documentos mais relevantes
    retrieved_docs = retriever.get_relevant_documents(query)
    
    # Obter a fonte do documento mais relevante
    source_files = list(set([doc.metadata["source"] for doc in retrieved_docs]))
    
    # Personalizar o prompt para respostas mais direcionadas se não encontrar a resposta em contexto
    custom_prompt = PromptTemplate(
        input_variables = ["context", "question"],
        template = """
        Você é um assistente da faculdade FACIMP Wyden que fornecerá informações sobre horários, salas, calendário
        acadêmico e lugares para resolução de problemas. Use apenas as informações abaixo para responder:
        
        Contexto: {context}

        Pergunta: {question}
        
        Se a resposta não estiver no conteúdo desse template, responda: "Peço desculpas, não posso lhe ajudar com isso."
        """
    )
    
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,
                                     chain_type_kwargs = {"prompt": custom_prompt})
    resposta = qa.run(query)
    auxiliar = 0
    # Adicionar referência do arquivo à resposta
    referencia = f"\n\n📄 **Fonte:** {', '.join(source_files)}"
    pattern = "Peço desculpas, não posso lhe ajudar com isso."
    if not re.search(pattern=pattern, string=resposta, flags=0):
        resposta += referencia
        auxiliar = 1
    
    # Adicionar ao histórico
    st.session_state.chat_history.append({"pergunta": query, "resposta": resposta})

# Exibir histórico com feedback moderno usando st.feedback
for idx, chat in enumerate(st.session_state.chat_history):
    with st.chat_message("user"):
        st.markdown(chat['pergunta'])
    with st.chat_message("assistant"):
        st.markdown(chat['resposta'])
        
        # Mecanismo de feedback nativo do Streamlit
        st.feedback(
            key=f"feedback_{idx}",
            on_change=lambda i=idx: save_feedback(i),
        )

# Botão para limpar histórico
if st.button("Limpar Conversa"):
    st.session_state.chat_history = []
    st.rerun()