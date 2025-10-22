from fastapi import APIRouter, HTTPException, status, UploadFile, File, Query, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from graphs.leadsGraph import leads_graph, LeadsState
from knowledge.pdfParser import pdf_parser
from rag.chunking import chunking_service
from vectorStore.embeddings import embedding_service
from vectorStore.qdrantClient import qdrant_service
from datetime import datetime
from middleware.auth import jwt_auth
from db.erpDb import get_entity_name

router = APIRouter(prefix="/api/leads", tags=["Leads"])

# Request/Response Models
class ChatRequest(BaseModel):
    entity: str = Field(..., description="Entity identifier (e.g., 'college_mit', 'school_dps')")
    query: str = Field(..., description="User's question")

class RetrievedDocument(BaseModel):
    score: float
    text: str
    url: Optional[str] = None
    title: Optional[str] = None

class ChatResponse(BaseModel):
    success: bool
    answer: str
    entity: str
    query: str
    retrieved_docs_count: int
    retrieved_docs: List[RetrievedDocument]
    timestamp: str

class PDFUploadResponse(BaseModel):
    success: bool
    message: str
    entity: str
    filename: str
    pages_count: int
    chunks_created: int
    vectors_stored: int
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    service: str

# Core chat function that can be called directly
async def process_chat(entity: str, query: str, model_name: str = None) -> Dict[str, Any]:
    """
    Core chat processing logic that can be reused.
    Returns dict with answer, retrieved_docs, entity, and query.

    Args:
        entity: Entity identifier
        query: User's question
        model_name: Model name from chatBotConfig (e.g., 'gpt-3.5-turbo', 'claude-sonnet', 'gemini-pro')
                    Falls back to DEFAULT_MODEL if not provided
    """
    import time
    from graphs.leadsGraph import get_llm_instance
    from config.settings import DEFAULT_MODEL

    start_time = time.time()

    # Validate inputs
    if not entity or not entity.strip():
        raise ValueError("Entity cannot be empty")

    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    validation_time = time.time()
    print(f"⏱️  Validation time: {validation_time - start_time:.3f}s")

    # Use default model if not provided
    if not model_name:
        model_name = DEFAULT_MODEL
        print(f"Using default model: {model_name}")
    else:
        print(f"Using model from chatBotConfig: {model_name}")

    # Initialize LLM instance based on model_name
    llm_init_start = time.time()
    llm_instance = get_llm_instance(model_name)
    llm_init_end = time.time()
    print(f"⏱️  LLM initialization time: {llm_init_end - llm_init_start:.3f}s")

    # Fetch entity name from entities collection
    entity_name = get_entity_name(entity)
    print(f"Entity name: {entity_name}")

    # Initialize state with model_name and llm_instance
    initial_state: LeadsState = {
        "messages": [],
        "entity": entity,
        "entity_name": entity_name,
        "query": query,
        "retrieved_docs": [],
        "answer": "",
        "model_name": model_name,
        "llm_instance": llm_instance
    }

    state_init_time = time.time()
    print(f"⏱️  State initialization time: {state_init_time - validation_time:.3f}s")

    # Invoke graph
    print(f"Processing chat request for entity={entity}")
    graph_start = time.time()
    result = leads_graph.invoke(initial_state)
    graph_end = time.time()
    print(f"⏱️  Graph execution time: {graph_end - graph_start:.3f}s")

    # Extract results
    extract_start = time.time()
    answer = result.get("answer", "")
    retrieved_docs = result.get("retrieved_docs", [])

    # Format retrieved documents
    formatted_docs = []
    for doc in retrieved_docs:
        formatted_docs.append({
            "score": doc["score"],
            "text": doc["text"][:500] + "..." if len(doc["text"]) > 500 else doc["text"],
            "url": doc["metadata"].get("url"),
            "title": doc["metadata"].get("title")
        })

    format_end = time.time()
    print(f"⏱️  Response formatting time: {format_end - extract_start:.3f}s")

    total_time = time.time() - start_time
    print(f"⏱️  🎯 TOTAL REQUEST TIME: {total_time:.3f}s")

    return {
        "answer": answer,
        "retrieved_docs": formatted_docs,
        "entity": entity,
        "entity_name": entity_name,
        "query": query,
        "retrieved_docs_count": len(retrieved_docs)
    }

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with AI agent using entity-specific knowledge base

    This endpoint:
    1. Retrieves relevant documents from the vector DB filtered by entity
    2. Uses LangGraph agent to generate contextual responses
    3. Returns the answer along with source documents

    - **entity**: Entity identifier (e.g., "college_mit", "school_dps_rkpuram")
    - **query**: User's question about the entity
    """
    try:
        # Call core chat processing function
        result = await process_chat(request.entity, request.query)

        # Convert retrieved_docs to RetrievedDocument models
        formatted_docs = [
            RetrievedDocument(
                score=doc["score"],
                text=doc["text"],
                url=doc.get("url"),
                title=doc.get("title")
            )
            for doc in result["retrieved_docs"]
        ]

        return ChatResponse(
            success=True,
            answer=result["answer"],
            entity=result["entity"],
            query=result["query"],
            retrieved_docs_count=result["retrieved_docs_count"],
            retrieved_docs=formatted_docs,
            timestamp=datetime.utcnow().isoformat()
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat request: {str(e)}"
        )

@router.post("/uploadPdf", response_model=PDFUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_pdf(
    entity: str = Query(..., description="Entity identifier (e.g., 'college_mit')"),
    file: UploadFile = File(..., description="PDF file to upload"),
    auth_data: dict = Depends(jwt_auth.verify_token)
):
    """
    Upload PDF brochure and store in vector database

    **Authentication Required**: This endpoint requires a valid JWT token in the Authorization header.

    This endpoint:
    1. Extracts text from PDF file
    2. Chunks the content into sentence windows
    3. Generates embeddings for each sentence
    4. Stores in vector DB with source_type='pdf'

    - **entity**: Entity identifier (query param)
    - **file**: PDF file (form data)
    """
    try:
        # Validate inputs
        if not entity or not entity.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Entity cannot be empty"
            )

        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are supported"
            )

        print(f"Processing PDF upload: {file.filename} for entity={entity}")

        # Read file content
        file_content = await file.read()

        # Extract text from PDF
        pdf_result = pdf_parser.extract_text_from_file(file_content, file.filename)

        if not pdf_result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error extracting PDF: {pdf_result.get('error', 'Unknown error')}"
            )

        extracted_text = pdf_result["text"]
        pdf_metadata = pdf_result["metadata"]

        if not extracted_text or not extracted_text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No text content found in PDF"
            )

        print(f"Extracted {len(extracted_text)} characters from {pdf_metadata['num_pages']} pages")

        # Prepare document for chunking
        document = {
            "text": extracted_text,
            "metadata": {
                "source_type": "pdf",
                "entity": entity,
                "filename": file.filename,
                "num_pages": pdf_metadata["num_pages"],
                "title": pdf_metadata.get("title", file.filename)
            }
        }

        # Chunk document
        chunks = chunking_service.chunk_documents([document])
        print(f"Created {len(chunks)} chunks from PDF")

        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No chunks created from PDF content"
            )

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
        print(f"Generating embeddings for {len(texts)} chunks")
        embeddings = embedding_service.generate_embeddings(texts)

        # Store in vector database
        print(f"Storing vectors in collection for entity={entity}")
        point_ids = qdrant_service.store_vectors(
            texts=texts,
            embeddings=embeddings,
            metadata_list=metadata_list
        )
        print(f"Stored {len(point_ids)} vectors from PDF")

        return PDFUploadResponse(
            success=True,
            message=f"Successfully processed PDF and stored {len(point_ids)} vectors",
            entity=entity,
            filename=file.filename,
            pages_count=pdf_metadata["num_pages"],
            chunks_created=len(chunks),
            vectors_stored=len(point_ids),
            timestamp=datetime.utcnow().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in PDF upload endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing PDF: {str(e)}"
        )

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service="leads"
    )
