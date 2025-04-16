# Usa uma imagem oficial do Python 3.12 como base
FROM python:3.12-slim

# Define o diretório de trabalho
WORKDIR /app


# Instala o UV (gerenciador de pacotes)
RUN pip install uv

# Copia os arquivos do projeto para dentro do container
COPY . .

# Instala as dependências a partir do pyproject.toml
RUN uv pip install --system .

# Expõe a porta padrão do Streamlit
EXPOSE 8501

# Define a variável de ambiente para o Streamlit não usar o navegador do container
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV PYTHONUNBUFFERED=1

# Comando para iniciar a aplicação Streamlit
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
