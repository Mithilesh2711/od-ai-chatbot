from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Dict, Any, Optional
from knowledge.scraper import web_scraper
from rag.chunking import chunking_service
from vectorStore.embeddings import embedding_service
from vectorStore.qdrantClient import qdrant_service
from datetime import datetime

router = APIRouter(prefix="/api/scraping", tags=["Scraping"])

# Request/Response Models
class EntityUrlScrapingRequest(BaseModel):
    url: HttpUrl = Field(..., description="URL to scrape (will scrape entire domain)")
    entity: str = Field(..., description="Entity identifier (e.g., 'college_xyz')")
    session: str = Field(..., description="Session identifier")
    max_pages: Optional[int] = Field(50, description="Maximum pages to scrape", ge=1, le=200)
    max_depth: Optional[int] = Field(3, description="Maximum depth to follow links", ge=1, le=5)

class ScrapingMetadata(BaseModel):
    pages_scraped: int
    chunks_created: int
    vectors_stored: int
    entity: str
    session: str
    source_type: str = "web_scrape"
    timestamp: str

class EntityUrlScrapingResponse(BaseModel):
    success: bool
    message: str
    metadata: ScrapingMetadata

class ScrapingStatusResponse(BaseModel):
    status: str
    message: str

async def scrape_and_store_task(
    url: str,
    entity: str,
    session: str,
    max_pages: int,
    max_depth: int
) -> Dict[str, Any]:
    """
    Background task to scrape website and store in vector DB

    Args:
        url: URL to scrape
        entity: Entity identifier
        session: Session identifier
        max_pages: Maximum pages to scrape
        max_depth: Maximum depth to follow links

    Returns:
        Metadata about the scraping operation
    """
    # Configure scraper
    web_scraper.max_pages = max_pages
    web_scraper.max_depth = max_depth

    # Scrape domain
    print(f"Starting scrape for {url}")
    scraped_pages = await web_scraper.scrape_domain(url)
    print(f"Scraped {len(scraped_pages)} pages")

    if not scraped_pages:
        raise Exception("No pages were successfully scraped")

    # Prepare documents for chunking
    documents = []
    for page in scraped_pages:
        if page.get("text") and page["text"].strip():
            documents.append({
                "text": page["text"],
                "metadata": {
                    "source_type": "web_scrape",
                    "entity": entity,
                    "session": session,
                    "url": page["url"],
                    "title": page.get("title", ""),
                    "depth": page.get("depth", 0)
                }
            })

    print(f"Prepared {len(documents)} documents for chunking")

    # Chunk documents
    all_chunks = chunking_service.chunk_documents(documents)
    print(f"Created {len(all_chunks)} chunks")

    if not all_chunks:
        raise Exception("No chunks were created from scraped content")

    # Prepare for vector storage
    texts = []
    metadata_list = []

    for chunk in all_chunks:
        texts.append(chunk["text"])

        # Combine chunk metadata with document metadata
        chunk_metadata = chunk["metadata"].copy()
        chunk_metadata["chunk_index"] = chunk["chunk_index"]
        chunk_metadata["total_chunks"] = chunk["total_chunks"]

        metadata_list.append(chunk_metadata)

    # Generate embeddings
    print(f"Generating embeddings for {len(texts)} chunks")
    embeddings = embedding_service.generate_embeddings(texts)
    print(f"Generated {len(embeddings)} embeddings")

    # Store in vector database (all go to same collection)
    print(f"Storing vectors in collection for entity={entity}, session={session}")
    point_ids = qdrant_service.store_vectors(
        texts=texts,
        embeddings=embeddings,
        metadata_list=metadata_list
    )
    print(f"Stored {len(point_ids)} vectors")

    return {
        "success": True,
        "pages_scraped": len(scraped_pages),
        "chunks_created": len(all_chunks),
        "vectors_stored": len(point_ids),
        "entity": entity,
        "session": session
    }

@router.post("/entityUrlScraping", response_model=EntityUrlScrapingResponse, status_code=status.HTTP_201_CREATED)
async def entity_url_scraping(request: EntityUrlScrapingRequest):
    """
    Scrape entity URL and store in vector database

    This endpoint will:
    1. Scrape the entire domain starting from the provided URL
    2. Extract and clean text content from all pages
    3. Chunk the content into smaller pieces
    4. Generate embeddings for each chunk
    5. Store embeddings in Qdrant vector database

    - **url**: URL to scrape (will crawl entire domain)
    - **entity**: Entity identifier (e.g., "college_mit", "school_dps")
    - **session**: Session identifier for tracking
    - **max_pages**: Maximum number of pages to scrape (default: 50)
    - **max_depth**: Maximum depth to follow links (default: 3)
    """
    try:
        # Validate inputs
        if not request.entity or not request.entity.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Entity cannot be empty"
            )

        if not request.session or not request.session.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Session cannot be empty"
            )

        # Execute scraping and storage
        result = await scrape_and_store_task(
            url=str(request.url),
            entity=request.entity,
            session=request.session,
            max_pages=request.max_pages,
            max_depth=request.max_depth
        )

        # Create metadata response
        metadata = ScrapingMetadata(
            pages_scraped=result["pages_scraped"],
            chunks_created=result["chunks_created"],
            vectors_stored=result["vectors_stored"],
            entity=result["entity"],
            session=result["session"],
            source_type="web_scrape",
            timestamp=datetime.utcnow().isoformat()
        )

        return EntityUrlScrapingResponse(
            success=True,
            message=f"Successfully scraped {result['pages_scraped']} pages and stored {result['vectors_stored']} vectors",
            metadata=metadata
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during scraping: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "scraping"}
