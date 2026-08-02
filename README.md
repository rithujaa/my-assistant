# Ask About Rithujaa: RAG-Powered Personal Profile Chatbot

**Live App:** https://rithujaa-ai-assistant.streamlit.app/

An AI-powered chatbot that answers questions about Rithujaa Rajendrakumar using Retrieval Augmented Generation (RAG). Built to demonstrate real-world RAG implementation on a personal knowledge base.

## What It Does

Ask anything about Rithujaa in plain English and get accurate, context-aware answers pulled directly from her profile documents:

- Work experience at Sonline LLC and Trilemma Foundation
- Projects (AI BI Assistant, Real-Time CDC Pipeline, DataBridge, and more)
- Skills and tools (Python, SQL, Snowflake, LLMs, Azure, AWS)
- Education (NYU MS Data Science, UBC BS Data Science)
- Background, interests, and what she is looking for in a role

## How It Works

1. Six structured documents about Rithujaa are loaded on startup
2. Documents are chunked using LangChain's RecursiveCharacterTextSplitter
3. Each chunk is embedded using OpenAI text-embedding-ada-002
4. Embeddings are stored in ChromaDB (in-memory, ephemeral)
5. When you ask a question, the most semantically relevant chunks are retrieved
6. GPT-4o-mini generates a natural, conversational answer from the retrieved context

## Tech Stack

- GPT-4o-mini (OpenAI) — answer generation
- OpenAI text-embedding-ada-002 — document embeddings
- LangChain — text splitting and retrieval
- ChromaDB — in-memory vector database
- Streamlit — web interface
- Python

## How to Run Locally

1. Clone the repo
2. Create and activate a virtual environment
3. Run pip install -r requirements.txt
4. Create a .env file with your OPENAI_API_KEY
5. Run streamlit run app.py
