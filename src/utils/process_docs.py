from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, chunk_size: int, overlap: int):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def load_docs(self, data_path: str) -> list[Document]:
        if not data_path:
            raise ValueError("data_path cannot be empty. Please provide a valid PDF file path.")

        document_loader = PDFPlumberLoader(data_path)
        data = document_loader.load()
        return data

    def split_docs(self, docs: list[Document]) -> list[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            length_function=len,
            is_separator_regex=False
        )

        return text_splitter.split_documents(docs)

    def process(self, path: str):
        docs = self.load_docs(path)
        return self.split_docs(docs)