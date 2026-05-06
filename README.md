# MedTech RAG System

A retrieval-augmented generation (RAG) assistant for MedTech troubleshooting documentation. This application demonstrates a complete pipeline for processing PDF manuals, building an embeddings vector store with Chroma, retrieving relevant context, and generating natural language answers via an LLM.

## Key Features

- Document ingestion from a PDF knowledge base (`data/raw/medtech_documents.pdf`).
- Chunk-based semantic search using sentence-transformer embeddings.
- Persistent vector store managed by Chroma.
- Context-aware prompt generation and response generation through Ollama.
- Interactive Gradio chat UI for asking troubleshooting questions.
- Conversation history support for follow-up context.

## Architecture
![Rag System Architecture](/img/rag-architecture.png)

The system consists of three main layers:

1. `src/utils/process_docs.py`
   - Loads PDF files with `PDFPlumberLoader`.
   - Splits large documents into overlapping semantic chunks.

2. `src/utils/populate_db.py`
   - Creates and manages a Chroma vector store.
   - Stores chunk embeddings and performs similarity search.

3. `src/engine/retrieve.py` + `src/engine/generate.py`
   - Retrieves top-k relevant chunks for a user query.
   - Builds a RAG prompt with retrieved context and chat history.
   - Generates answers using the configured LLM.

The end-to-end pipeline is orchestrated by `src/rag_pipeline.py` and exposed through a browser-based chat interface in `src/app.py`.

## Requirements

- Python 3.11
- Dependencies are listed in `requirements.txt`

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure the source PDF exists at `data/raw/medtech_documents.pdf`.

4. Install Ollama

- Follow the official installation instructions:
    - Visit: https://ollama.com/download
    - Choose your operating system (macOS, Linux, or Windows)
    - Download and run the installer

- Pull the Mistral Model
```bash
ollama pull mistral
```

## Usage

Run the app from the `src` directory:

```bash
python src/app.py
```

This launches a Gradio web interface where you can enter troubleshooting questions and view retrieved context alongside model responses.

![Rag User Interface](/img/rag-ui.jpeg)

## Project Structure

```
project-root/ 
│
├── data/ 
│ ├── raw/ 
│ │ └── medtech_documents.pdf
│ └── chroma_db/ 
│
├── config.py
│
├── src/
│   ├── app.py 
│   ├── rag_pipeline.py 
│   ├── config.py 
│   ├── engine/               
│   │   ├── generate.py
│   │   └── retrieve.py
│   └── utils/             
│       └── populate_db.py
│       └── process_docs.py
│
├── eval/ 
│ ├── evaluate.ipynb
│ └── output_images/ 
│
├── requirements.txt 
├── .gitignore
└── README.md 
```

- `src/app.py` — Gradio interface and chat routing
- `src/rag_pipeline.py` — pipeline orchestration and query flow
- `src/engine/generate.py` — prompt construction and LLM interaction
- `src/engine/retrieve.py` — similarity-based retrieval workflow
- `src/utils/process_docs.py` — PDF loading and text splitting
- `src/utils/populate_db.py` — Chroma vector store ingestion and query
- `data/` — raw documents and generated Chroma database files
- `eval/` — evaluation artifacts and notebooks

## License

This project is licensed under the MIT License.
