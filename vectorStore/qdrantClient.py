from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchParams
)
from typing import List, Dict, Any, Optional
from config import settings
import uuid
from datetime import datetime

class QdrantService:
    """Service for interacting with Qdrant vector database"""

    def __init__(self):
        self.url = settings.QDRANT_URL
        self.api_key = settings.QDRANT_API_KEY
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self.dimension = settings.EMBEDDING_DIMENSION
        self.client = None

    def _get_client(self) -> QdrantClient:
        """Get or create Qdrant client"""
        if self.client is None:
            self.client = QdrantClient(
                url=self.url,
                api_key=self.api_key
            )
        return self.client

    def create_collection(self, collection_name: Optional[str] = None):
        """
        Create a new collection in Qdrant with payload indexes

        Args:
            collection_name: Name of the collection (uses default if not provided)
        """
        client = self._get_client()
        col_name = collection_name or self.collection_name

        # Check if collection exists
        collections = client.get_collections().collections
        collection_exists = any(col.name == col_name for col in collections)

        if collection_exists:
            print(f"Collection '{col_name}' already exists")
            # Ensure indexes exist even if collection exists
            self._create_payload_indexes(col_name)
            return

        # Create collection with cosine similarity
        client.create_collection(
            collection_name=col_name,
            vectors_config=VectorParams(
                size=self.dimension,
                distance=Distance.COSINE
            )
        )
        print(f"Collection '{col_name}' created successfully")

        # Create payload indexes for filtering
        self._create_payload_indexes(col_name)

    def _create_payload_indexes(self, collection_name: str):
        """
        Create payload indexes for entity and session fields

        Args:
            collection_name: Name of the collection
        """
        client = self._get_client()

        try:
            # Create index for entity field
            client.create_payload_index(
                collection_name=collection_name,
                field_name="entity",
                field_schema="keyword"
            )
            print(f"Created payload index for 'entity' field")
        except Exception as e:
            print(f"Index for 'entity' may already exist: {e}")

        try:
            # Create index for session field
            client.create_payload_index(
                collection_name=collection_name,
                field_name="session",
                field_schema="keyword"
            )
            print(f"Created payload index for 'session' field")
        except Exception as e:
            print(f"Index for 'session' may already exist: {e}")

        try:
            # Create index for source_type field
            client.create_payload_index(
                collection_name=collection_name,
                field_name="source_type",
                field_schema="keyword"
            )
            print(f"Created payload index for 'source_type' field")
        except Exception as e:
            print(f"Index for 'source_type' may already exist: {e}")

    def store_vectors(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata_list: List[Dict[str, Any]],
        collection_name: Optional[str] = None
    ) -> List[str]:
        """
        Store vectors in Qdrant

        Args:
            texts: List of text chunks
            embeddings: List of embedding vectors
            metadata_list: List of metadata dicts for each chunk
            collection_name: Name of the collection (uses default if not provided)

        Returns:
            List of point IDs
        """
        client = self._get_client()
        col_name = collection_name or self.collection_name

        # Ensure collection exists
        self.create_collection(col_name)

        # Create points
        points = []
        point_ids = []

        for text, embedding, metadata in zip(texts, embeddings, metadata_list):
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)

            # Combine text and metadata
            payload = {
                "text": text,
                "timestamp": datetime.utcnow().isoformat(),
                **metadata
            }

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
            )

        # Upsert points in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            client.upsert(
                collection_name=col_name,
                points=batch
            )

        return point_ids

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filters: Optional metadata filters (e.g., {"entity": "college_xyz"})
            collection_name: Name of the collection (uses default if not provided)

        Returns:
            List of search results with text, metadata, and score
        """
        client = self._get_client()
        col_name = collection_name or self.collection_name

        # Build filter if provided
        query_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
            if conditions:
                query_filter = Filter(must=conditions)

        # Search
        results = client.search(
            collection_name=col_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True
        )

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result.id,
                "score": result.score,
                "text": result.payload.get("text", ""),
                "metadata": {k: v for k, v in result.payload.items() if k != "text"}
            })

        return formatted_results

    def delete_by_filter(
        self,
        filters: Dict[str, Any],
        collection_name: Optional[str] = None
    ):
        """
        Delete points by metadata filter

        Args:
            filters: Metadata filters (e.g., {"entity": "college_xyz", "session": "sess_123"})
            collection_name: Name of the collection (uses default if not provided)
        """
        client = self._get_client()
        col_name = collection_name or self.collection_name

        # Build filter
        conditions = []
        for key, value in filters.items():
            conditions.append(
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value)
                )
            )

        if conditions:
            client.delete(
                collection_name=col_name,
                points_selector=Filter(must=conditions)
            )

    def get_collection_info(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a collection

        Args:
            collection_name: Name of the collection (uses default if not provided)

        Returns:
            Collection info dictionary
        """
        client = self._get_client()
        col_name = collection_name or self.collection_name

        try:
            info = client.get_collection(collection_name=col_name)
            return {
                "name": col_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            return {
                "name": col_name,
                "error": str(e),
                "exists": False
            }

# Global instance
qdrant_service = QdrantService()
