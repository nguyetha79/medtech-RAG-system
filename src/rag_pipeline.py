import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'engine'))

from generate import LLMGenerator
from retrieve import Retriever
from populate_db import VectorStore
from process_docs import DocumentProcessor
import config

class RAGPipeline:
    def __init__(self):
        # Initialize DocumentProcessor
        self.doc_processor = DocumentProcessor(
            chunk_size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP
        )

        # Process documents and add to vector store
        print("Processing documents...")
        self.chunks = self.doc_processor.process(config.PDF_PATH)

        # Initialize VectorStore (will create or load the Chroma DB)
        self.vector_store = VectorStore(
            embedding_model_id=config.EMBEDDING_MODEL_ID,
            chroma_path=config.CHROMA_PATH,
            k=config.TOP_K
        )
        self.vector_store.add_docs_to_db(self.chunks)

        # Initialize Retriever
        self.retriever = Retriever(
            vector_store=self.vector_store
        )

        # Initialize LLMGenerator
        self.llm = LLMGenerator(
            model_name=config.MODEL_NAME,
            max_new_tokens=config.MAX_NEW_TOKENS,
            temperature=config.TEMPERATURE,
            rag_prompt_template=config.RAG_PROMPT_TEMPLATE
        )
        print("RAG Pipeline components initialized successfully!")

    def ask(self, query: str, history: list) -> tuple[str, list]:
        """Executes the RAG pipeline for a given query.
        Returns the answer and the retrieved contexts.
        """
        contexts = self.retriever.retrieve(query)
        prompt = self.llm.build_prompt(query, contexts, history)
        response = self.llm.generate(prompt)
        return response, contexts


