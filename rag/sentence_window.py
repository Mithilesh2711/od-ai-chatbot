"""
Sentence window retrieval module for RAG enhancement
Retrieves small chunks and expands with surrounding context
"""
from typing import List, Dict, Any, Optional
import time


class SentenceWindowService:
    """
    Service for sentence window retrieval and context expansion
    """

    def __init__(self, window_size: int = 3):
        """
        Initialize sentence window service

        Args:
            window_size: Number of sentences to include before and after
        """
        self.window_size = window_size

    def expand_context(
        self,
        documents: List[Dict[str, Any]],
        qdrant_service=None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Expand retrieved documents with surrounding context

        For documents with sentence window metadata, retrieves surrounding sentences.
        For documents without metadata, returns as-is (backward compatible).

        Args:
            documents: List of retrieved documents
            qdrant_service: Qdrant service instance for fetching related chunks
            filters: Filters to apply when fetching related chunks

        Returns:
            Documents with expanded context
        """
        if not documents:
            return documents

        expanded_docs = []

        for doc in documents:
            metadata = doc.get('metadata', {})

            # Check if this is a sentence window chunk
            if 'parent_id' in metadata and 'sentence_index' in metadata:
                # This is a sentence window chunk - expand it
                expanded_doc = self._expand_sentence_window(
                    doc=doc,
                    qdrant_service=qdrant_service,
                    filters=filters
                )
                expanded_docs.append(expanded_doc)
            else:
                # Not a sentence window chunk - return as-is (backward compatible)
                expanded_docs.append(doc)

        return expanded_docs

    def _expand_sentence_window(
        self,
        doc: Dict[str, Any],
        qdrant_service=None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Expand a single sentence window chunk with surrounding context

        Args:
            doc: Document with sentence window metadata
            qdrant_service: Qdrant service for fetching related chunks
            filters: Filters to apply

        Returns:
            Document with expanded context
        """
        metadata = doc.get('metadata', {})
        parent_id = metadata.get('parent_id')
        sentence_index = metadata.get('sentence_index')

        if not qdrant_service or parent_id is None or sentence_index is None:
            # Cannot expand - return original
            return doc

        try:
            # Fetch surrounding chunks from the same parent
            # This is a simplified approach - in practice, you might store
            # the full parent document and slice it, or fetch sibling chunks

            # For now, we'll use the stored window_before and window_after if available
            window_before = metadata.get('window_before', '')
            window_after = metadata.get('window_after', '')

            if window_before or window_after:
                # Construct expanded text
                expanded_text_parts = []
                if window_before:
                    expanded_text_parts.append(window_before)
                expanded_text_parts.append(doc['text'])  # The matched sentence
                if window_after:
                    expanded_text_parts.append(window_after)

                expanded_text = ' '.join(expanded_text_parts)

                # Create expanded document
                expanded_doc = {
                    **doc,
                    'text': expanded_text,
                    'original_text': doc['text'],  # Keep original sentence
                    'metadata': {
                        **metadata,
                        'context_expanded': True
                    }
                }

                return expanded_doc
            else:
                # No window context available - return original
                return doc

        except Exception as e:
            print(f"  ⚠️  [SENTENCE_WINDOW] Failed to expand context: {e}")
            return doc

    def split_into_sentence_windows(
        self,
        text: str,
        parent_id: str,
        window_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Split text into sentence-level chunks with window context

        This is a helper for data ingestion (not used in retrieval).

        Args:
            text: Full text to split
            parent_id: ID of the parent document
            window_size: Number of sentences to include before/after (uses default if not provided)

        Returns:
            List of sentence chunks with window metadata
        """
        window_size = window_size or self.window_size

        # Simple sentence splitting (can be improved with NLTK or spaCy)
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        for idx, sentence in enumerate(sentences):
            # Get surrounding sentences for context
            start_idx = max(0, idx - window_size)
            end_idx = min(len(sentences), idx + window_size + 1)

            window_before = ' '.join(sentences[start_idx:idx])
            window_after = ' '.join(sentences[idx + 1:end_idx])

            chunk = {
                'text': sentence,  # Store just the sentence for retrieval
                'metadata': {
                    'parent_id': parent_id,
                    'sentence_index': idx,
                    'total_sentences': len(sentences),
                    'window_before': window_before,
                    'window_after': window_after
                }
            }
            chunks.append(chunk)

        return chunks


# Global instance
sentence_window_service: Optional[SentenceWindowService] = None


def get_sentence_window_service(window_size: int = 3) -> SentenceWindowService:
    """
    Get or create global sentence window service instance

    Args:
        window_size: Number of sentences to include before and after

    Returns:
        SentenceWindowService instance
    """
    global sentence_window_service

    if sentence_window_service is None:
        sentence_window_service = SentenceWindowService(window_size=window_size)

    return sentence_window_service
