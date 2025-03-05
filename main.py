import streamlit as st
import os
import chromadb
import datetime
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

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
        loader = Docx2txtLoader(temp_path)
    else:
        st.error("Formato de arquivo não suportado")
        return []
    
    documents = loader.load()
    os.remove(temp_path)  # Remover arquivo temporário após uso
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    # Adicionar metadados com base no nome do arquivo
    for doc in docs:
        doc.metadata = {
            "source": file.name,
            "length": len(doc.page_content),
            "upload_date": str(datetime.datetime.now())
        }
    
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
            vectorstore.add_documents(docs)
        st.success("Todos os arquivos foram processados e indexados com sucesso!")

# Histórico de chat usando st.chat_message
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.subheader("Chat com os Documentos")
with st.chat_message("assistant"):
    st.markdown("Olá! Envie sua pergunta sobre os documentos carregados.")

query = st.chat_input("Digite sua pergunta...")
if query:
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, **LLM_PARAMS)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Buscar os 3 documentos mais relevantes
    retrieved_docs = retriever.get_relevant_documents(query)
    
    # Obter a fonte do documento mais relevante
    source_files = list(set([doc.metadata["source"] for doc in retrieved_docs]))
    
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    resposta = qa.run(query)
    
    # Adicionar referência do arquivo à resposta
    referencia = f"\n\n📄 **Fonte:** {', '.join(source_files)}"
    resposta += referencia
    
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
            on_change=lambda: st.toast("Obrigado pelo feedback!", icon="✅"),
        )

# Botão para limpar histórico
if st.button("Limpar Conversa"):
    st.session_state.chat_history = []
    st.rerun()
