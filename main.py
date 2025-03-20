import streamlit as st
import numpy as np
import os
import chromadb
import datetime
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configurar ChromaDB local
chroma_client = chromadb.PersistentClient(path="./chroma_db")
vectorstore = Chroma(
    collection_name="documents",
    embedding_function=OpenAIEmbeddings(
        model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY
    ),
    persist_directory="./chroma_db",
)

# Define um threshold mínimo de similaridade para considerar uma resposta válida
SIMILARITY_THRESHOLD = 0.7  # Ajuste conforme necessário

# Carregar modelo de embeddings
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

MENSAGENS_GENERICAS = [
    "oi",
    "olá",
    "bom dia",
    "boa tarde",
    "boa noite",
    "e aí",
    "hello",
    "como vai",
    "tudo bem",
    "fala",
]


def is_generic_message(user_message, threshold=0.85):
    """Verifica se a mensagem do usuário é similar a uma mensagem genérica"""
    user_embedding = np.array(embedding_model.embed_query(user_message)).reshape(1, -1)
    generic_embeddings = np.array(
        [embedding_model.embed_query(msg) for msg in MENSAGENS_GENERICAS]
    )

    # Calcular similaridade de cosseno
    similarities = cosine_similarity(user_embedding, generic_embeddings)

    # Retorna True se alguma similaridade for maior que o threshold
    return np.max(similarities) >= threshold


# Criar função para processar arquivos com metadados
def process_file(file):
    ext = file.name.split(".")[-1]
    temp_path = f"./temp_uploaded.{ext}"

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
    os.remove(temp_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,  # Chunks maiores para evitar cortes ruins
        chunk_overlap=200,  # Sobreposição para manter contexto
        separators=["\n\n", ".", "?"],  # Evita cortes no meio de frases
    )
    docs = text_splitter.split_documents(documents)

    for doc in docs:
        doc.metadata = {
            "source": file.name,
            "length": len(doc.page_content),
            "upload_date": str(datetime.datetime.now()),
        }

    return docs


# Interface Streamlit
st.set_page_config(page_title="Chatbot RAG - RH", layout="wide")
st.title("Chatbot RAG para RH")

# Sidebar para upload de múltiplos arquivos
with st.sidebar:
    st.header("Upload de Documentos")
    uploaded_files = st.file_uploader(
        "Faça upload de PDFs ou Word", type=["pdf", "docx"], accept_multiple_files=True
    )
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
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature=0.2,
        max_tokens=512,
        model_kwargs={"top_p": 0.95, "frequency_penalty": 0.0, "presence_penalty": 0.0},
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Uso no chatbot:
    if is_generic_message(query.lower()):
        resposta = "Olá! Como posso te ajudar hoje?"
    else:
        # 🔹 **Passo 2: Recuperar documentos relevantes e verificar similaridade**
        retrieved_docs = retriever.get_relevant_documents(query)
        relevancy_scores = [
            doc.metadata.get("similarity", 1.0) for doc in retrieved_docs
        ]  # Assume 1.0 se não houver metadata

        if any(score >= SIMILARITY_THRESHOLD for score in relevancy_scores):
            qa = RetrievalQA.from_chain_type(
                llm=llm, chain_type="stuff", retriever=retriever
            )
            resposta = qa.run(query)

            # Obtém as fontes corretas dos documentos usados
            source_files = list(set([doc.metadata["source"] for doc in retrieved_docs]))

            # 🔹 **Nova verificação para garantir que a resposta veio dos documentos**
            documentos_conteudo = [doc.page_content for doc in retrieved_docs]

            if any(resposta in doc_content for doc_content in documentos_conteudo):
                resposta += f"\n\n📄 **Fonte:** {', '.join(source_files)}"

        else:
            # **Se não houver documentos relevantes, mensagem genérica SEM fonte**
            resposta = "No momento, não fui atualizado com essas informações. Gostaria de tentar outra pergunta?"

    # Armazena histórico
    st.session_state.chat_history.append({"pergunta": query, "resposta": resposta})

# Exibir histórico com feedback moderno usando st.feedback
for idx, chat in enumerate(st.session_state.chat_history):
    with st.chat_message("user"):
        st.markdown(chat["pergunta"])
    if chat["resposta"]:  # Apenas exibir a resposta se houver conteúdo
        with st.chat_message("assistant"):
            st.markdown(chat["resposta"])

            st.feedback(
                key=f"feedback_{idx}",
                on_change=lambda: st.toast("Obrigado pelo feedback!", icon="✅"),
            )

# Botão para limpar histórico
if st.button("Limpar Conversa"):
    st.session_state.chat_history = []
    st.rerun()
