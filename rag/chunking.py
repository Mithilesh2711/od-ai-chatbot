from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import tiktoken

class ChunkingService:
    """Service for chunking documents into smaller pieces"""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        encoding_name: str = "cl100k_base"
    ):
        """
        Initialize chunking service

        Args:
            chunk_size: Target size of each chunk in tokens
            chunk_overlap: Number of tokens to overlap between chunks
            encoding_name: Tokenizer encoding (cl100k_base for GPT-4/3.5)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=encoding_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Chunk a single text document

        Args:
            text: The text to chunk
            metadata: Optional metadata to attach to each chunk

        Returns:
            List of chunks with text and metadata
        """
        if not text or not text.strip():
            return []

        # Split text into chunks
        chunks = self.text_splitter.split_text(text)

        # Create chunk objects with metadata
        chunk_objects = []
        for idx, chunk in enumerate(chunks):
            chunk_obj = {
                "text": chunk,
                "chunk_index": idx,
                "total_chunks": len(chunks),
                "metadata": metadata or {}
            }
            chunk_objects.append(chunk_obj)

        return chunk_objects

    def chunk_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents

        Args:
            documents: List of documents with 'text' and 'metadata' keys

        Returns:
            List of all chunks from all documents
        """
        all_chunks = []

        for doc in documents:
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})

            chunks = self.chunk_text(text, metadata)
            all_chunks.extend(chunks)

        return all_chunks

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text

        Args:
            text: The text to count tokens for

        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))

# Default chunking service instance
chunking_service = ChunkingService()
