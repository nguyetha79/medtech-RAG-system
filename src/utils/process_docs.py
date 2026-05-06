from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    """Processor for loading and splitting documents

    Handles PDF loading using PDFPlumberLoader and text splitting
    with RecursiveCharacterTextSplitter
    """

    def __init__(self, chunk_size: int, overlap: int):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def load_docs(self, data_path: str) -> list[Document]:
        """Load documents from a PDF file

        Args:
            data_path (str): Path to the PDF file

        Returns:
            list[Document]: List of loaded documents

        Raises:
            ValueError: If data_path is empty
        """
        if not data_path:
            raise ValueError("data_path cannot be empty. Please provide a valid PDF file path.")

        document_loader = PDFPlumberLoader(data_path)
        data = document_loader.load()

        return data

    def split_docs(self, docs: list[Document]) -> list[Document]:
        """Split documents into smaller chunks

        Args:
            docs (list[Document]): List of documents to split

        Returns:
            list[Document]: List of split document chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            length_function=len,
            is_separator_regex=False
        )

        return text_splitter.split_documents(docs)

    def process(self, path: str):
        """Process a PDF file by loading and splitting it

        Args:
            path (str): Path to the PDF file

        Returns:
            list[Document]: List of processed document chunks
        """
        docs = self.load_docs(path)
        return self.split_docs(docs)