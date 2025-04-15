import streamlit as st
import yaml
import os
from dotenv import load_dotenv
from streamlit_authenticator import Authenticate
from yaml.loader import SafeLoader
from functions import *

load_dotenv() 

vectorstore = vector_store()
st.set_page_config(layout="wide")
col1, col2 = st.columns(2)

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

config["credentials"]["usernames"]["adm"]["password"] = os.getenv("ADMIN_PASS")

authenticator = Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

login_placeholder = st.empty()
with login_placeholder.container():
    col1, col2 = st.columns(2)
    with col1:
        st.image("facimp_logo.png", use_container_width=True)
    with col2:
        authenticator.login()

auth_status = st.session_state.get("authentication_status")

if auth_status:
    login_placeholder.empty()  # limpa o formulário
    st.success(f'Bem vindo, {st.session_state["name"]}')
    # Sidebar para upload de múltiplos arquivos
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
    authenticator.logout()
elif auth_status is False:
    st.error("Usuário ou senha inválido.")
elif auth_status is None:
    st.warning("Por favor, preencha com seu usuário e senha.")
