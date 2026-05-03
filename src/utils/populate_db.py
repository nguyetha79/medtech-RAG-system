from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

class VectorStore:
    def __init__(self, embedding_model_id: str, chroma_path: str, k: int):
        self.embedding_model_id = embedding_model_id
        self.chroma_path = chroma_path
        self.k = k
        self.embedding_function = self._get_embedding_function()
        self.vector_db = self._create_chroma_db()

    def _get_embedding_function(self):
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_id
        )
        
        return embeddings

    def _create_chroma_db(self):
        return Chroma(
            persist_directory=self.chroma_path,
            embedding_function=self.embedding_function
        )

    def _calculate_chunk_ids(self, chunks: list[Document]) -> list[Document]:
        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            current_page_id = f"{source}:{page}"

            # If the page ID is the same as the last one, increment the index.
            # Otherwise, reset the index for a new page.
            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id

            chunk.metadata["id"] = chunk_id

        return chunks

    def add_docs_to_db(self, chunks: list[Document]):
        chunks_with_ids = self._calculate_chunk_ids(chunks)

        try:
            existing_items = self.vector_db.get(include=[])
            existing_ids = set(existing_items["ids"])
            print(f"Number of existing documents in DB: {len(existing_ids)}")
        except Exception as e:
            print(f"Could not retrieve existing documents, assuming empty DB: {e}")
            existing_ids = set()

        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        if len(new_chunks):
            print(f"Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            self.vector_db.add_documents(new_chunks, ids=new_chunk_ids)
        else:
            print("No new documents to add")

    def query_db(self, query_text: str) -> list[tuple[Document, float]]:
        """Queries the vector database for relevant documents and their scores."""
        results = self.vector_db.similarity_search_with_score(query_text, k=self.k)
        return results

    def clear_db(self):
        if self.vector_db and self.vector_db.get(include=[])["ids"]:
            # Only delete if the collection actually exists and has items
            self.vector_db.delete_collection()
            print("Chroma database collection cleared.")
        else:
            print("Chroma database is empty or not initialized.")
