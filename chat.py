import streamlit as st
import os
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import re
from functions import *

# Carregar variáveis de ambiente
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

vectorstore = vector_store()

init_db()

# Parâmetros da LLM definidos estaticamente
LLM_PARAMS = {
    "temperature": 0.7,
    "max_tokens": 512,
    "top_p": 0.95,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
}

# Interface Streamlit
st.set_page_config(page_title="Chatbot RAG", layout="wide")
st.title("Chatbot RAG com Streamlit e ChromaDB")

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
        Porém, se você receber cumprimentos, responda de acordo com o cumprimento e adicione: "Como posso te ajudar hoje?".
        """
    )
    
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,
                                     chain_type_kwargs = {"prompt": custom_prompt})
    resposta = qa.run(query)
    auxiliar = 0
    # Adicionar referência do arquivo à resposta
    referencia = f"\n\n📄 **Fonte:** {', '.join(source_files)}"
    patterns = ["Peço desculpas.*", "Como posso.*ajudar"]
    if not any(re.search(p, resposta) for p in patterns):
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