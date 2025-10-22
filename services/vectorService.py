from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from vectorStore.embeddings import embedding_service
from vectorStore.qdrantClient import qdrant_service
from middleware.auth import jwt_auth

router = APIRouter(prefix="/api/vector", tags=["Vector"])

# Request/Response Models
class StoreVectorRequest(BaseModel):
    texts: List[str] = Field(..., description="List of text chunks to store")
    metadata_list: List[Dict[str, Any]] = Field(..., description="List of metadata for each chunk")
    collection_name: Optional[str] = Field(None, description="Collection name (uses default if not provided)")

class StoreVectorResponse(BaseModel):
    success: bool
    message: str
    point_ids: List[str]
    count: int

class SearchRequest(BaseModel):
    query: str = Field(..., description="Query text to search for")
    top_k: int = Field(5, description="Number of results to return", ge=1, le=100)
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters (e.g., {'entity': 'college_xyz'})")
    collection_name: Optional[str] = Field(None, description="Collection name (uses default if not provided)")

class SearchResult(BaseModel):
    id: str
    score: float
    text: str
    metadata: Dict[str, Any]

class SearchResponse(BaseModel):
    success: bool
    results: List[SearchResult]
    count: int

class DeleteRequest(BaseModel):
    filters: Dict[str, Any] = Field(..., description="Metadata filters for deletion (e.g., {'entity': 'college_xyz'})")
    collection_name: Optional[str] = Field(None, description="Collection name (uses default if not provided)")

class DeleteResponse(BaseModel):
    success: bool
    message: str

class CollectionInfoResponse(BaseModel):
    success: bool
    info: Dict[str, Any]

@router.post("/store", response_model=StoreVectorResponse, status_code=status.HTTP_201_CREATED)
async def store_vectors(request: StoreVectorRequest):
    """
    Store text embeddings in vector database

    - **texts**: List of text chunks to embed and store
    - **metadata_list**: List of metadata dicts (must match length of texts)
    - **collection_name**: Optional collection name
    """
    try:
        # Validate input
        if len(request.texts) != len(request.metadata_list):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Length of texts and metadata_list must match"
            )

        if not request.texts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="texts list cannot be empty"
            )

        # Generate embeddings
        embeddings = embedding_service.generate_embeddings(request.texts)

        # Store in vector database
        point_ids = qdrant_service.store_vectors(
            texts=request.texts,
            embeddings=embeddings,
            metadata_list=request.metadata_list,
            collection_name=request.collection_name
        )

        return StoreVectorResponse(
            success=True,
            message=f"Successfully stored {len(point_ids)} vectors",
            point_ids=point_ids,
            count=len(point_ids)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error storing vectors: {str(e)}"
        )

@router.post("/search", response_model=SearchResponse)
async def search_vectors(request: SearchRequest):
    """
    Search for similar vectors using semantic search

    - **query**: Text query to search for
    - **top_k**: Number of results to return (1-100)
    - **filters**: Optional metadata filters (e.g., {"entity": "college_xyz"})
    - **collection_name**: Optional collection name
    """
    try:
        # Generate query embedding
        query_embedding = embedding_service.generate_embedding(request.query)

        # Search vector database
        results = qdrant_service.search(
            query_embedding=query_embedding,
            top_k=request.top_k,
            filters=request.filters,
            collection_name=request.collection_name
        )

        # Format response
        search_results = [
            SearchResult(
                id=result["id"],
                score=result["score"],
                text=result["text"],
                metadata=result["metadata"]
            )
            for result in results
        ]

        return SearchResponse(
            success=True,
            results=search_results,
            count=len(search_results)
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching vectors: {str(e)}"
        )

@router.delete("/delete", response_model=DeleteResponse)
async def delete_vectors(request: DeleteRequest):
    """
    Delete vectors by metadata filter

    - **filters**: Metadata filters for deletion (e.g., {"entity": "college_xyz", "session": "sess_123"})
    - **collection_name**: Optional collection name
    """
    try:
        if not request.filters:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="filters cannot be empty"
            )

        # Delete from vector database
        qdrant_service.delete_by_filter(
            filters=request.filters,
            collection_name=request.collection_name
        )

        return DeleteResponse(
            success=True,
            message=f"Successfully deleted vectors matching filters: {request.filters}"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting vectors: {str(e)}"
        )

@router.delete("/deleteEntity/{entity}", response_model=DeleteResponse)
async def delete_entity_vectors(
    entity: str,
    collection_name: Optional[str] = None,
    auth_data: dict = Depends(jwt_auth.verify_token)
):
    """
    Delete all vectors for a specific entity

    **Authentication Required**: This endpoint requires a valid JWT token in the Authorization header.

    This endpoint will delete all vectors (from web scraping, PDF uploads, etc.)
    associated with the specified entity.

    - **entity**: Entity identifier to delete all vectors for (e.g., "college_mit", "school_dps")
    - **collection_name**: Optional collection name (uses default if not provided)
    """
    try:
        if not entity or not entity.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Entity cannot be empty"
            )

        # Delete all vectors with this entity
        filters = {"entity": entity}

        qdrant_service.delete_by_filter(
            filters=filters,
            collection_name=collection_name
        )

        return DeleteResponse(
            success=True,
            message=f"Successfully deleted all vectors for entity: {entity}"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting entity vectors: {str(e)}"
        )

@router.get("/collections/{collection_name}", response_model=CollectionInfoResponse)
async def get_collection_info(collection_name: str):
    """
    Get information about a collection

    - **collection_name**: Name of the collection
    """
    try:
        info = qdrant_service.get_collection_info(collection_name)

        return CollectionInfoResponse(
            success=True,
            info=info
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching collection info: {str(e)}"
        )

@router.post("/create-indexes/{collection_name}")
async def create_indexes(collection_name: str):
    """
    Manually create payload indexes for a collection

    - **collection_name**: Name of the collection to create indexes for
    """
    try:
        qdrant_service._create_payload_indexes(collection_name)
        return {
            "success": True,
            "message": f"Indexes created for collection '{collection_name}'"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating indexes: {str(e)}"
        )

@router.post("/recreate-collection/{collection_name}")
async def recreate_collection(collection_name: str):
    """
    Delete and recreate collection with current embedding dimensions
    ⚠️ WARNING: This deletes all data in the collection!

    - **collection_name**: Name of the collection to recreate
    """
    try:
        client = qdrant_service._get_client()

        # Delete if exists
        try:
            client.delete_collection(collection_name=collection_name)
            print(f"Deleted existing collection '{collection_name}'")
        except Exception:
            print(f"Collection '{collection_name}' doesn't exist, creating new")

        # Create with new dimensions
        qdrant_service.create_collection(collection_name)

        return {
            "success": True,
            "message": f"Collection '{collection_name}' recreated with {qdrant_service.dimension} dimensions",
            "dimension": qdrant_service.dimension
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error recreating collection: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "vector"}
