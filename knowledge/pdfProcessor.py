"""
PDF Processing Utility
Handles PDF processing from both file uploads and URLs
"""
from typing import Dict, Any, List
from knowledge.pdfParser import pdf_parser
from rag.chunking import chunking_service
from vectorStore.embeddings import embedding_service
from vectorStore.qdrantClient import qdrant_service
import httpx


async def download_pdf_from_url(url: str, timeout: int = 30) -> bytes:
    """
    Download PDF from URL

    Args:
        url: PDF URL
        timeout: Request timeout in seconds

    Returns:
        PDF file content as bytes

    Raises:
        Exception: If download fails
    """
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()

            # Verify content type is PDF
            content_type = response.headers.get("content-type", "")
            if "pdf" not in content_type.lower():
                # Still try if URL ends with .pdf
                if not url.lower().endswith(".pdf"):
                    raise Exception(f"URL does not return PDF content (content-type: {content_type})")

            return response.content

        except httpx.HTTPError as e:
            raise Exception(f"Failed to download PDF from {url}: {str(e)}")


async def process_pdf_to_vectors(
    file_content: bytes,
    filename: str,
    entity: str,
    url: str = None
) -> Dict[str, Any]:
    """
    Process PDF and store in vector database.
    Can be used for both file uploads and URL downloads.

    Args:
        file_content: PDF file content as bytes
        filename: Name of the PDF file
        entity: Entity identifier
        url: Optional URL where PDF was downloaded from

    Returns:
        Dictionary with processing results and metadata
    """
    try:
        print(f"[PDF_PROCESSOR] Processing PDF: {filename} for entity={entity}")

        # Extract text from PDF
        pdf_result = pdf_parser.extract_text_from_file(file_content, filename)

        if not pdf_result["success"]:
            return {
                "success": False,
                "error": f"Error extracting PDF: {pdf_result.get('error', 'Unknown error')}",
                "filename": filename,
                "entity": entity
            }

        extracted_text = pdf_result["text"]
        pdf_metadata = pdf_result["metadata"]

        if not extracted_text or not extracted_text.strip():
            return {
                "success": False,
                "error": "No text content found in PDF",
                "filename": filename,
                "entity": entity
            }

        print(f"[PDF_PROCESSOR] Extracted {len(extracted_text)} characters from {pdf_metadata['num_pages']} pages")

        # Prepare document for chunking
        document = {
            "text": extracted_text,
            "metadata": {
                "source_type": "pdf",
                "entity": entity,
                "filename": filename,
                "num_pages": pdf_metadata["num_pages"],
                "title": pdf_metadata.get("title", filename),
                "url": url or f"upload://{filename}"  # Use provided URL or mark as upload
            }
        }

        # Chunk document
        chunks = chunking_service.chunk_documents([document])
        print(f"[PDF_PROCESSOR] Created {len(chunks)} chunks from PDF")

        if not chunks:
            return {
                "success": False,
                "error": "No chunks created from PDF content",
                "filename": filename,
                "entity": entity
            }

        # Prepare for vector storage
        texts = []
        metadata_list = []

        for chunk in chunks:
            texts.append(chunk["text"])

            # Combine chunk metadata
            chunk_metadata = chunk["metadata"].copy()
            chunk_metadata["chunk_index"] = chunk["chunk_index"]
            chunk_metadata["total_chunks"] = chunk["total_chunks"]

            metadata_list.append(chunk_metadata)

        # Generate embeddings
        print(f"[PDF_PROCESSOR] Generating embeddings for {len(texts)} chunks")
        embeddings = embedding_service.generate_embeddings(texts)

        # Store in vector database
        print(f"[PDF_PROCESSOR] Storing vectors in collection for entity={entity}")
        point_ids = qdrant_service.store_vectors(
            texts=texts,
            embeddings=embeddings,
            metadata_list=metadata_list
        )
        print(f"[PDF_PROCESSOR] ✓ Stored {len(point_ids)} vectors from PDF")

        return {
            "success": True,
            "filename": filename,
            "entity": entity,
            "pages_count": pdf_metadata["num_pages"],
            "chunks_created": len(chunks),
            "vectors_stored": len(point_ids),
            "url": url
        }

    except Exception as e:
        print(f"[PDF_PROCESSOR] ERROR processing PDF {filename}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "filename": filename,
            "entity": entity
        }


async def process_pdf_from_url(url: str, entity: str) -> Dict[str, Any]:
    """
    Download PDF from URL and process it to vectors.

    Args:
        url: PDF URL
        entity: Entity identifier

    Returns:
        Dictionary with processing results
    """
    try:
        # Extract filename from URL
        filename = url.split("/")[-1].split("?")[0]  # Remove query params
        if not filename.endswith(".pdf"):
            filename = f"{filename}.pdf"

        print(f"[PDF_PROCESSOR] Downloading PDF from URL: {url}")

        # Download PDF
        file_content = await download_pdf_from_url(url)
        print(f"[PDF_PROCESSOR] Downloaded {len(file_content)} bytes")

        # Process PDF
        result = await process_pdf_to_vectors(
            file_content=file_content,
            filename=filename,
            entity=entity,
            url=url
        )

        return result

    except Exception as e:
        print(f"[PDF_PROCESSOR] ERROR processing PDF from URL {url}: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "url": url,
            "entity": entity
        }
