"""
Reranking module for RAG enhancement using Cohere or fallback methods
"""
from typing import List, Dict, Any, Optional
import time


class RerankerService:
    """
    Service for reranking retrieved documents using Cohere API
    Falls back to score-based sorting if Cohere is unavailable
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "rerank-english-v3.0"):
        """
        Initialize reranker service

        Args:
            api_key: Cohere API key (optional)
            model: Reranking model name
        """
        self.api_key = api_key
        self.model = model
        self.cohere_client = None

        # Try to initialize Cohere if API key is provided
        if api_key:
            try:
                import cohere
                self.cohere_client = cohere.Client(api_key)
                print(f"✓ Cohere reranker initialized with model: {model}")
            except ImportError:
                print("⚠ Cohere library not installed. Install with: pip install cohere")
                print("  Falling back to score-based reranking")
            except Exception as e:
                print(f"⚠ Failed to initialize Cohere: {e}")
                print("  Falling back to score-based reranking")

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance to query

        Args:
            query: User query
            documents: List of retrieved documents with 'text', 'score', 'metadata'
            top_n: Number of top documents to return

        Returns:
            Reranked list of documents with updated scores
        """
        if not documents:
            return documents

        start_time = time.time()

        # Try Cohere reranking first
        if self.cohere_client:
            try:
                reranked_docs = self._rerank_with_cohere(query, documents, top_n)
                elapsed = time.time() - start_time
                print(f"  ⏱️  [RERANK] Cohere reranking: {elapsed:.3f}s")
                return reranked_docs
            except Exception as e:
                print(f"⚠ Cohere reranking failed: {e}")
                print("  Falling back to score-based reranking")

        # Fallback to score-based reranking
        reranked_docs = self._rerank_by_score(documents, top_n)
        elapsed = time.time() - start_time
        print(f"  ⏱️  [RERANK] Score-based reranking: {elapsed:.3f}s")
        return reranked_docs

    def _rerank_with_cohere(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: int
    ) -> List[Dict[str, Any]]:
        """
        Rerank using Cohere's rerank API

        Args:
            query: User query
            documents: List of documents
            top_n: Number of top documents to return

        Returns:
            Reranked documents with Cohere relevance scores
        """
        # Extract text from documents for Cohere
        texts = [doc['text'] for doc in documents]

        # Call Cohere rerank API
        response = self.cohere_client.rerank(
            query=query,
            documents=texts,
            top_n=top_n,
            model=self.model
        )

        # Map results back to original documents with new scores
        reranked_docs = []
        for result in response.results:
            original_doc = documents[result.index]
            reranked_doc = {
                **original_doc,
                'score': result.relevance_score,  # Cohere relevance score
                'original_score': original_doc['score'],  # Keep original vector similarity score
                'rerank_method': 'cohere'
            }
            reranked_docs.append(reranked_doc)

        return reranked_docs

    def _rerank_by_score(
        self,
        documents: List[Dict[str, Any]],
        top_n: int
    ) -> List[Dict[str, Any]]:
        """
        Fallback reranking: simply sort by existing vector similarity score

        Args:
            documents: List of documents
            top_n: Number of top documents to return

        Returns:
            Top-N documents sorted by score
        """
        # Sort by score (descending) and take top_n
        sorted_docs = sorted(documents, key=lambda x: x['score'], reverse=True)[:top_n]

        # Add rerank method indicator
        for doc in sorted_docs:
            doc['rerank_method'] = 'score_fallback'

        return sorted_docs


# Global instance (initialized later with config)
reranker_service: Optional[RerankerService] = None


def get_reranker_service(api_key: Optional[str] = None, model: str = "rerank-english-v3.0") -> RerankerService:
    """
    Get or create global reranker service instance

    Args:
        api_key: Cohere API key
        model: Reranking model name

    Returns:
        RerankerService instance
    """
    global reranker_service

    if reranker_service is None:
        reranker_service = RerankerService(api_key=api_key, model=model)

    return reranker_service
