from langchain_openai import OpenAIEmbeddings
from typing import List
from config import settings

class EmbeddingService:
    """Service for generating embeddings using OpenAI"""

    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        self.dimension = settings.EMBEDDING_DIMENSION
        self.embeddings = OpenAIEmbeddings(
            model=self.model_name
        )

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding vector
        """
        embedding = self.embeddings.embed_query(text)
        return embedding

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batched for efficiency)

        Args:
            texts: List of input texts to embed

        Returns:
            List of embedding vectors
        """
        embeddings = self.embeddings.embed_documents(texts)
        return embeddings

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors"""
        return self.dimension

# Global instance
embedding_service = EmbeddingService()
