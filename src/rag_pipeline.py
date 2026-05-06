import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'engine'))

from generate import LLMGenerator
from retrieve import Retriever
from populate_db import VectorStore
from process_docs import DocumentProcessor
import config

class RAGPipeline:
    """A pipeline for Retrieval-Augmented Generation (RAG) in medtech applications

    This class integrates document processing, vector storage, retrieval, and language model generation
    to provide context-aware answers to queries based on processed documents.
    """
    
    def __init__(self):
        # Initialize DocumentProcessor
        self.doc_processor = DocumentProcessor(
            chunk_size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP
        )

        # Process documents and add to vector store
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
        """Execute the RAG pipeline for a given query

        Args:
            query (str): The input query to process
            history (list): The conversation history or prior context

        Returns:
            tuple[str, list]: A tuple containing the generated answer (str)
                and the retrieved contexts (list)
        """

        contexts = self.retriever.retrieve(query)
        prompt = self.llm.build_prompt(query, contexts, history)
        response = self.llm.generate(prompt)

        return response, contexts


