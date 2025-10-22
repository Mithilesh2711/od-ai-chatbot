"""
Sentence window chunking service for RAG
Replaces traditional chunking with sentence-level retrieval
"""
from typing import List, Dict, Any
import re
import uuid


class ChunkingService:
    """Service for chunking documents using sentence window approach"""

    def __init__(
        self,
        window_size: int = 3,
    ):
        """
        Initialize sentence window chunking service

        Args:
            window_size: Number of sentences to include before and after for context
        """
        self.window_size = window_size

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Split on sentence boundaries (.!?) followed by whitespace
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Clean up empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def chunk_text(
        self,
        text: str,
        metadata: Dict[str, Any] = None,
        parent_id: str = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk text into sentence windows

        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk
            parent_id: Parent document ID (generated if not provided)

        Returns:
            List of sentence window chunks
        """
        if not text or not text.strip():
            return []

        # Generate parent ID if not provided
        if parent_id is None:
            parent_id = str(uuid.uuid4())

        # Split into sentences
        sentences = self._split_into_sentences(text)

        if not sentences:
            return []

        chunks = []
        for idx, sentence in enumerate(sentences):
            # Get surrounding sentences for context window
            start_idx = max(0, idx - self.window_size)
            end_idx = min(len(sentences), idx + self.window_size + 1)

            # Build context windows
            window_before = ' '.join(sentences[start_idx:idx]) if idx > 0 else ''
            window_after = ' '.join(sentences[idx + 1:end_idx]) if idx < len(sentences) - 1 else ''

            # Create chunk with sentence window metadata
            chunk_metadata = {
                **(metadata or {}),
                'parent_id': parent_id,
                'sentence_index': idx,
                'total_sentences': len(sentences),
                'window_before': window_before,
                'window_after': window_after
            }

            chunk = {
                'text': sentence,  # Store only the sentence for embedding
                'chunk_index': idx,
                'total_chunks': len(sentences),
                'metadata': chunk_metadata
            }
            chunks.append(chunk)

        return chunks

    def chunk_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents using sentence window approach

        Args:
            documents: List of documents with 'text' and 'metadata' keys

        Returns:
            List of all sentence window chunks from all documents
        """
        all_chunks = []

        for doc in documents:
            text = doc.get('text', '')
            metadata = doc.get('metadata', {})

            # Generate unique parent ID for each document
            parent_id = str(uuid.uuid4())

            chunks = self.chunk_text(text, metadata, parent_id)
            all_chunks.extend(chunks)

        return all_chunks


# Default chunking service instance
chunking_service = ChunkingService(window_size=3)
