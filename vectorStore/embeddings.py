from langchain_openai import OpenAIEmbeddings
from typing import List
from config import settings
import hashlib
import time
from functools import lru_cache

class EmbeddingCache:
    """Simple in-memory cache for embeddings with TTL"""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 1800):
        """
        Initialize embedding cache

        Args:
            max_size: Maximum number of cached embeddings
            ttl_seconds: Time-to-live for cache entries (default: 30 minutes)
        """
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0

    def _get_hash(self, text: str) -> str:
        """Generate hash for text"""
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, text: str) -> List[float]:
        """
        Get embedding from cache if available and not expired

        Args:
            text: Input text

        Returns:
            Cached embedding or None
        """
        text_hash = self._get_hash(text)

        if text_hash in self.cache:
            # Check if expired
            if time.time() - self.timestamps[text_hash] < self.ttl_seconds:
                self.hits += 1
                return self.cache[text_hash]
            else:
                # Expired - remove
                del self.cache[text_hash]
                del self.timestamps[text_hash]

        self.misses += 1
        return None

    def set(self, text: str, embedding: List[float]):
        """
        Store embedding in cache

        Args:
            text: Input text
            embedding: Embedding vector
        """
        text_hash = self._get_hash(text)

        # Simple eviction: if cache is full, clear oldest 20%
        if len(self.cache) >= self.max_size:
            sorted_items = sorted(self.timestamps.items(), key=lambda x: x[1])
            to_remove = int(self.max_size * 0.2)
            for hash_key, _ in sorted_items[:to_remove]:
                del self.cache[hash_key]
                del self.timestamps[hash_key]

        self.cache[text_hash] = embedding
        self.timestamps[text_hash] = time.time()

    def get_stats(self) -> dict:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "cache_size": len(self.cache)
        }


class EmbeddingService:
    """Service for generating embeddings using OpenAI with caching and connection pooling"""

    def __init__(self, enable_cache: bool = True):
        """
        Initialize embedding service

        Args:
            enable_cache: Whether to enable embedding cache
        """
        self.model_name = settings.EMBEDDING_MODEL
        self.dimension = settings.EMBEDDING_DIMENSION

        # Initialize cache
        self.enable_cache = enable_cache
        if enable_cache:
            self.cache = EmbeddingCache(max_size=1000, ttl_seconds=1800)

        # Initialize embeddings (OpenAI handles connection pooling internally)
        self.embeddings = OpenAIEmbeddings(
            model=self.model_name,
            dimensions=self.dimension
        )
        print(f"✓ Embedding service initialized with {self.dimension}D vectors")

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text (with caching)

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding vector
        """
        # Check cache first
        if self.enable_cache:
            cached = self.cache.get(text)
            if cached is not None:
                return cached

        # Generate embedding
        embedding = self.embeddings.embed_query(text)

        # Store in cache
        if self.enable_cache:
            self.cache.set(text, embedding)

        return embedding

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batched for efficiency, with caching)

        Args:
            texts: List of input texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Check cache for each text
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        if self.enable_cache:
            for idx, text in enumerate(texts):
                cached = self.cache.get(text)
                if cached is not None:
                    results[idx] = cached
                else:
                    uncached_indices.append(idx)
                    uncached_texts.append(text)
        else:
            uncached_indices = list(range(len(texts)))
            uncached_texts = texts

        # Generate embeddings for uncached texts (batched)
        if uncached_texts:
            new_embeddings = self.embeddings.embed_documents(uncached_texts)

            # Store new embeddings in cache and results
            for idx, embedding in zip(uncached_indices, new_embeddings):
                results[idx] = embedding
                if self.enable_cache:
                    self.cache.set(texts[idx], embedding)

        return results

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors"""
        return self.dimension

    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        if self.enable_cache:
            return self.cache.get_stats()
        return {"cache_enabled": False}

# Global instance
# Cache is enabled by default, controlled by settings
try:
    from config import settings
    embedding_service = EmbeddingService(enable_cache=settings.ENABLE_EMBEDDING_CACHE)
except ImportError:
    # Fallback if settings not available
    embedding_service = EmbeddingService(enable_cache=True)
