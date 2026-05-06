"""
Configuration module for the MedTech RAG system

Defines file paths, system constants, and prompt templates used
throughout the application
"""

PDF_PATH: str = "data/raw/medtech_documents.pdf"
CHROMA_PATH: str = "data/chroma_db"

EMBEDDING_MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME: str = "mistral"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K: int = 5

MAX_NEW_TOKENS: int = 256
TEMPERATURE: float = 0.3

RAG_PROMPT_TEMPLATE: str = """\
    You are a helpful assistant. Answer the question using ONLY the information \
    provided in the context below.
    If the context does not contain enough information, say "I don't know."

    Chat History:
    {chat_history}

    Context:
    {context}

    Question: {question}

    Answer:"""