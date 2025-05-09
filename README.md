# 📚 Chatbot RAG com Streamlit e ChromaDB

Este é um projeto de chatbot baseado em **Retrieval-Augmented Generation (RAG)**, utilizando **Streamlit** para interface, **ChromaDB** como banco vetorial e **LangChain** para integração com modelos de linguagem (LLM).

## 🛠 Tecnologias Utilizadas

- **Python 3.12**
- **Streamlit** (Frontend + Backend)
- **ChromaDB** (Banco de dados vetorial)
- **LangChain** (Integração com LLMs)
- **OpenAI API** (Para gerar respostas)
- **PyPDFLoader / Docx2txtLoader** (Para processar PDFs e documentos do Word)
- **UV (gerenciador de ambiente Python)**

## 🚀 Configuração do Ambiente com UV

Este projeto usa **UV** como gerenciador de pacotes, que é mais rápido e eficiente que o pip tradicional.

### 1️⃣ **Instalar o UV (se ainda não tiver)**

```sh
pip install uv
```

### 2️⃣ **Criar e sincronizar o ambiente virtual**

```sh

uv sync

source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

## 🔑 Configuração da OpenAI API

O projeto utiliza a **API da OpenAI**, então você precisa configurar sua chave de API.

1. **Crie um arquivo **``** na raiz do projeto**
2. **Adicione sua chave de API no arquivo:**

```sh
OPENAI_API_KEY="sua-chave-aqui"
```

## 🏃‍♂️ Rodando o Projeto

Depois de configurar o ambiente e instalar as dependências, execute o seguinte comando para iniciar o chatbot:

```sh
streamlit run main.py
```

O chatbot estará disponível no navegador no endereço [**http://localhost:8501**](http://localhost:8501).

## ✅ Funcionalidades

- ✅ Upload de arquivos **PDF e DOCX**
- ✅ Indexação de documentos com **ChromaDB**
- ✅ Geração de respostas inteligentes usando **OpenAI GPT**
- ✅ Interface moderna com **Streamlit**
- ✅ **Mecanismo de feedback** para avaliar respostas
- ✅ Histórico de conversa salvo na sessão

## 💡 Melhorias Futuras

- Em progresso

## 📌 Contribuição

Se quiser contribuir, sinta-se à vontade para abrir **issues** e enviar **pull requests**!

## 📜 Licença

Este projeto é de código aberto e pode ser utilizado conforme sua necessidade.

---

Agora você está pronto para rodar e expandir seu **Chatbot RAG com Streamlit e ChromaDB**! 🚀🎯