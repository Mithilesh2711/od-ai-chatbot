from fastapi import APIRouter, HTTPException, status, BackgroundTasks, Depends
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Dict, Any, Optional
from knowledge.scraper import web_scraper
from knowledge.pdfProcessor import process_pdf_from_url
from rag.chunking import chunking_service
from vectorStore.embeddings import embedding_service
from vectorStore.qdrantClient import qdrant_service
from datetime import datetime
from middleware.auth import jwt_auth
from services.requestLogService import request_log_service
import uuid

router = APIRouter(prefix="/api/scraping", tags=["Scraping"])

# Request/Response Models
class EntityUrlScrapingRequest(BaseModel):
    url: HttpUrl = Field(..., description="URL to scrape (will scrape entire domain)")
    entity: str = Field(..., description="Entity identifier (e.g., 'college_xyz')")
    max_pages: Optional[int] = Field(100, description="Maximum pages to scrape", ge=1, le=200)
    max_depth: Optional[int] = Field(10, description="Maximum depth to follow links", ge=1, le=5)

class ScrapingMetadata(BaseModel):
    pages_scraped: int
    pdfs_processed: int = 0
    chunks_created: int
    vectors_stored: int
    entity: str
    source_type: str = "web_scrape"
    timestamp: str

class EntityUrlScrapingResponse(BaseModel):
    success: bool
    message: str
    status: str
    entity: str
    estimated_time: str
    metadata: Optional[ScrapingMetadata] = None

class ScrapingStatusResponse(BaseModel):
    status: str
    message: str

async def scrape_and_store_task(
    url: str,
    entity: str,
    max_pages: int,
    max_depth: int,
    session_id: str = None,
    user_id: str = None,
    user_name: str = None,
    user_mobile: str = None
) -> Dict[str, Any]:
    """
    Background task to scrape website and store in vector DB
    Also detects and processes PDF URLs during scraping

    Args:
        url: URL to scrape
        entity: Entity identifier
        max_pages: Maximum pages to scrape
        max_depth: Maximum depth to follow links
        session_id: Session identifier for logging
        user_id: User ID for logging
        user_name: User name for logging
        user_mobile: User mobile for logging

    Returns:
        Metadata about the scraping operation
    """
    try:
        print(f"[BACKGROUND] Starting scrape task for entity={entity}, url={url}")

        # Configure scraper
        web_scraper.max_pages = max_pages
        web_scraper.max_depth = max_depth

        # Scrape domain (returns both pages and PDF URLs)
        print(f"[BACKGROUND] Starting scrape for {url}")
        scraped_pages, pdf_urls_found = await web_scraper.scrape_domain(url)
        print(f"[BACKGROUND] Scraped {len(scraped_pages)} pages, found {len(pdf_urls_found)} PDFs")

        # Track PDF processing
        pdfs_processed = 0
        total_vectors_from_pdfs = 0

        if not scraped_pages and not pdf_urls_found:
            print(f"[BACKGROUND] ERROR: No pages or PDFs were found for {url}")
            return {
                "success": False,
                "error": "No pages or PDFs were found",
                "entity": entity
            }

        # Process PDFs first if found
        if pdf_urls_found:
            print(f"[BACKGROUND] Processing {len(pdf_urls_found)} PDF files...")
            for pdf_url in pdf_urls_found:
                try:
                    pdf_result = await process_pdf_from_url(pdf_url, entity)
                    if pdf_result.get("success"):
                        pdfs_processed += 1
                        total_vectors_from_pdfs += pdf_result.get("vectors_stored", 0)
                        print(f"[BACKGROUND] ✓ PDF processed: {pdf_url} ({pdf_result.get('vectors_stored', 0)} vectors)")
                    else:
                        print(f"[BACKGROUND] ✗ Failed to process PDF: {pdf_url} - {pdf_result.get('error', 'Unknown error')}")
                except Exception as e:
                    print(f"[BACKGROUND] ✗ Error processing PDF {pdf_url}: {str(e)}")

        # Prepare documents for chunking (regular web pages)
        documents = []
        if scraped_pages:
            for page in scraped_pages:
                if page.get("text") and page["text"].strip():
                    documents.append({
                        "text": page["text"],
                        "metadata": {
                            "source_type": "web_scrape",
                            "entity": entity,
                            "url": page["url"],
                            "title": page.get("title", ""),
                            "depth": page.get("depth", 0)
                        }
                    })

            print(f"[BACKGROUND] Prepared {len(documents)} web pages for chunking")

        # Chunk documents
        all_chunks = []
        if documents:
            all_chunks = chunking_service.chunk_documents(documents)
            print(f"[BACKGROUND] Created {len(all_chunks)} chunks from web pages")

        # Prepare for vector storage (web pages)
        total_vectors_from_pages = 0
        if all_chunks:
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
            print(f"[BACKGROUND] Generating embeddings for {len(texts)} chunks")
            embeddings = embedding_service.generate_embeddings(texts)
            print(f"[BACKGROUND] Generated {len(embeddings)} embeddings")

            # Store in vector database (all go to same collection)
            print(f"[BACKGROUND] Storing vectors in collection for entity={entity}")
            point_ids = qdrant_service.store_vectors(
                texts=texts,
                embeddings=embeddings,
                metadata_list=metadata_list
            )
            total_vectors_from_pages = len(point_ids)
            print(f"[BACKGROUND] ✓ Stored {len(point_ids)} vectors from web pages")

        # Calculate totals
        total_vectors = total_vectors_from_pages + total_vectors_from_pdfs
        total_chunks = len(all_chunks) + total_vectors_from_pdfs  # Approximate

        print(f"[BACKGROUND] ✓ Scraping completed:")
        print(f"  - Pages: {len(scraped_pages)}")
        print(f"  - PDFs: {pdfs_processed}/{len(pdf_urls_found)}")
        print(f"  - Total vectors: {total_vectors}")

        # Log the scraping operation to Qdrant
        if session_id:
            try:
                request_log_service.log_request(
                    entity=entity,
                    session=session_id,
                    operation_type="web_scraping",
                    user_id=user_id,
                    user_name=user_name,
                    user_mobile=user_mobile,
                    pdf_url=None,
                    website_url=url,
                    metadata={
                        "pages_scraped": len(scraped_pages),
                        "pdfs_processed": pdfs_processed,
                        "total_pdfs_found": len(pdf_urls_found),
                        "chunks_created": total_chunks,
                        "vectors_stored": total_vectors,
                        "max_pages": max_pages,
                        "max_depth": max_depth,
                        "status": "success"
                    }
                )
                print(f"[BACKGROUND] ✓ Request logged successfully")
            except Exception as log_error:
                print(f"[BACKGROUND] Warning: Failed to log request: {str(log_error)}")

        return {
            "success": True,
            "pages_scraped": len(scraped_pages),
            "pdfs_processed": pdfs_processed,
            "chunks_created": total_chunks,
            "vectors_stored": total_vectors,
            "entity": entity
        }

    except Exception as e:
        print(f"[BACKGROUND] ERROR in scraping task for entity={entity}: {str(e)}")
        import traceback
        print(f"[BACKGROUND] Traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e),
            "entity": entity
        }

@router.post("/entityUrlScraping", response_model=EntityUrlScrapingResponse, status_code=status.HTTP_202_ACCEPTED)
async def entity_url_scraping(
    request: EntityUrlScrapingRequest,
    background_tasks: BackgroundTasks,
    auth_data: dict = Depends(jwt_auth.verify_token)
):
    """
    Scrape entity URL and store in vector database (Async - returns immediately)

    **Authentication Required**: This endpoint requires a valid JWT token in the Authorization header.

    This endpoint will start scraping in the background and return immediately:
    1. Scrape the entire domain starting from the provided URL
    2. Extract and clean text content from all pages
    3. Chunk the content into sentence windows
    4. Generate embeddings for each sentence
    5. Store embeddings in Qdrant vector database

    **Note**: This is an asynchronous operation. The endpoint returns immediately with status "processing".
    The actual scraping happens in the background and may take several minutes depending on the website size.

    - **url**: URL to scrape (will crawl entire domain)
    - **entity**: Entity identifier (e.g., "college_mit", "school_dps")
    - **max_pages**: Maximum number of pages to scrape (default: 100)
    - **max_depth**: Maximum depth to follow links (default: 10)
    """
    try:
        # Validate inputs
        if not request.entity or not request.entity.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Entity cannot be empty"
            )

        # Calculate estimated time based on max_pages (rough estimate: 2 seconds per page)
        estimated_minutes = (request.max_pages * 2) // 60
        if estimated_minutes < 1:
            estimated_time = "1-2 minutes"
        else:
            estimated_time = f"{estimated_minutes}-{estimated_minutes + 2} minutes"

        # Generate session ID for logging
        session_id = str(uuid.uuid4())
        user_doc = auth_data.get("user", {})

        # Add scraping task to background
        background_tasks.add_task(
            scrape_and_store_task,
            url=str(request.url),
            entity=request.entity,
            max_pages=request.max_pages,
            max_depth=request.max_depth,
            session_id=session_id,
            user_id=auth_data.get("user_id"),
            user_name=user_doc.get("name", ""),
            user_mobile=user_doc.get("mobile", "")
        )

        print(f"Background scraping task started for entity, url={request.url}")

        return EntityUrlScrapingResponse(
            success=True,
            message=f"Scraping task started for entity. This will run in the background.",
            status="processing",
            entity=request.entity,
            estimated_time=estimated_time,
            metadata=None
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting scraping task: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "scraping"}
