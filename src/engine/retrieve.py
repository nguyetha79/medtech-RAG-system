import sys
import os

from typing import NamedTuple

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from populate_db import VectorStore

class RetrievedChunk(NamedTuple):
    """A single retrieved document chunk with its relevance score."""
    rank: int
    score: float
    text: str

class Retriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Retrieves relevant document chunks from the vector store."""
        results_with_scores = self.vector_store.query_db(query_text=query)

        retrieved_chunks = []
        for rank, (doc, distance) in enumerate(results_with_scores):
            similarity = 1 - distance

            retrieved_chunks.append(
                RetrievedChunk(
                    rank=rank + 1,
                    score=float(similarity),
                    text=doc.page_content,
                )
            )
        return retrieved_chunks
