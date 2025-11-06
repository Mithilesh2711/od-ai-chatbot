from fastapi import APIRouter, HTTPException, status, UploadFile, File, Query, Depends
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Dict, Any, Optional
from graphs.leadsGraph import leads_graph, LeadsState
from knowledge.pdfProcessor import process_pdf_to_vectors, process_pdf_from_url
from datetime import datetime
from middleware.auth import jwt_auth
from db.erpDb import get_entity_name
from services.translationService import with_translation
from services.requestLogService import request_log_service
from config.settings import ENABLE_MULTILINGUAL
import uuid

router = APIRouter(prefix="/api/leads", tags=["Leads"])

# Request/Response Models
class ChatRequest(BaseModel):
    entity: str = Field(..., description="Entity identifier (e.g., 'college_mit', 'school_dps')")
    query: str = Field(..., description="User's question")
    # Optional webhook fields from ERP
    fromNumber: Optional[str] = Field(None, description="User phone number (for webhook)")
    toNumber: Optional[str] = Field(None, description="Bot phone number (for webhook)")
    accessToken: Optional[str] = Field(None, description="mTalkz access token (for webhook)")
    model: Optional[str] = Field(None, description="Chatbot model name (for webhook)")

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

class PDFUrlRequest(BaseModel):
    """Request model for PDF URL upload"""
    url: HttpUrl = Field(..., description="PDF URL to download and process")
    entity: str = Field(..., description="Entity identifier (e.g., 'college_mit')")

class PDFUploadResponse(BaseModel):
    success: bool
    message: str
    entity: str
    filename: str
    pages_count: int
    chunks_created: int
    vectors_stored: int
    url: Optional[str] = None
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    service: str

# Core chat function that can be called directly (without translation)
async def _process_chat_core(entity: str, query: str, model_name: str = None, phone_number: str = None, llm_instance=None) -> Dict[str, Any]:
    """
    Core chat processing logic that can be reused.
    Returns dict with answer, retrieved_docs, entity, and query.

    Args:
        entity: Entity identifier
        query: User's question (should be in English when called via translation middleware)
        model_name: Model name from chatBotConfig (e.g., 'gpt-3.5-turbo', 'claude-sonnet', 'gemini-pro')
                    Falls back to DEFAULT_MODEL if not provided
        phone_number: User's phone number for session tracking (used for thread_id)
        llm_instance: Optional pre-initialized LLM instance (for translation middleware)
    """
    import time
    from graphs.leadsGraph import get_llm_instance
    from config.settings import DEFAULT_MODEL, ENABLE_CHAT_MEMORY

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

    # Initialize LLM instance based on model_name (if not already provided)
    llm_init_start = time.time()
    if not llm_instance:
        llm_instance = get_llm_instance(model_name)
        print(f"⏱️  LLM initialization time: {time.time() - llm_init_start:.3f}s")
    else:
        print(f"⏱️  Using pre-initialized LLM instance")

    # Fetch entity name from entities collection
    entity_name = get_entity_name(entity)
    print(f"Entity name: {entity_name}")

    # Initialize state with model_name and llm_instance
    # IMPORTANT: Don't set "messages" key to allow checkpointer to restore from memory
    initial_state: LeadsState = {
        "entity": entity,
        "entity_name": entity_name,
        "query": query,
        "query_type": None,  # Will be set by classify node
        "confidence_score": None,  # Will be set by classify node
        "query_variations": None,
        "hypothetical_doc": None,
        "retrieved_docs": [],
        "answer": "",
        "needs_rag": None,  # Will be determined by classify node
        "model_name": model_name,
        "llm_instance": llm_instance
    }

    state_init_time = time.time()
    print(f"⏱️  State initialization time: {state_init_time - validation_time:.3f}s")

    # Create config for thread-based memory (if enabled and phone_number provided)
    config = None
    if ENABLE_CHAT_MEMORY and phone_number:
        # Generate thread_id based on entity and phone number
        thread_id = f"{entity}_{phone_number}"
        config = {"configurable": {"thread_id": thread_id}}
        print(f"💬 [MEMORY] Using thread_id: {thread_id}")
    elif ENABLE_CHAT_MEMORY and not phone_number:
        print(f"⚠️  [MEMORY] Chat memory enabled but no phone_number provided - using stateless mode")

    # Invoke graph
    print(f"Processing chat request for entity={entity}")
    graph_start = time.time()
    if config:
        result = leads_graph.invoke(initial_state, config)
    else:
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


# Wrapped version with translation middleware
@with_translation(enabled=ENABLE_MULTILINGUAL)
async def process_chat(entity: str, query: str, model_name: str = None, phone_number: str = None, **kwargs) -> Dict[str, Any]:
    """
    Chat processing with multilingual support (Hindi, English, Hinglish).

    This function wraps _process_chat_core with translation middleware:
    1. Detects input language (Hindi/English/Hinglish)
    2. Translates query to English if needed
    3. Processes in English (RAG + LLM)
    4. Translates response back to detected language

    Args:
        entity: Entity identifier
        query: User's question in any supported language
        model_name: Model name from chatBotConfig
        phone_number: User's phone number for session tracking
        **kwargs: Additional args (llm_instance passed by translation middleware)

    Returns:
        Dict with answer (in original language), retrieved_docs, entity, query, etc.
    """
    # Get llm_instance from kwargs (initialized by translation middleware)
    llm_instance = kwargs.get('llm_instance')

    # Call core function with llm_instance
    return await _process_chat_core(
        entity=entity,
        query=query,
        model_name=model_name,
        phone_number=phone_number,
        llm_instance=llm_instance
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with AI agent using entity-specific knowledge base

    This endpoint:
    1. Retrieves relevant documents from the vector DB filtered by entity
    2. Uses LangGraph agent to generate contextual responses
    3. Returns the answer along with source documents
    4. If webhook fields provided (fromNumber, toNumber, accessToken), sends WhatsApp message

    - **entity**: Entity identifier (e.g., "college_mit", "school_dps_rkpuram")
    - **query**: User's question about the entity
    - **fromNumber** (optional): User phone for webhook + thread_id
    - **toNumber** (optional): Bot phone for webhook
    - **accessToken** (optional): mTalkz token for webhook
    - **model** (optional): Model name for webhook
    """
    try:
        # Check if this is a webhook request (has fromNumber, toNumber, accessToken)
        is_webhook = bool(request.fromNumber and request.toNumber and request.accessToken)

        # Call core chat processing function with model and phone if provided
        result = await process_chat(
            entity=request.entity,
            query=request.query,
            model_name=request.model,
            phone_number=request.fromNumber
        )

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

        # If webhook request, send WhatsApp message via mtalkzService
        if is_webhook:
            try:
                from services.mtalkzService import send_interactive_button_message

                answer = result["answer"]
                body_text = answer[:1000] if len(answer) > 1000 else answer

                buttons = [{"id": "main_menu", "title": "Main Menu"}]

                await send_interactive_button_message(
                    to_phone=request.toNumber,
                    from_phone=request.fromNumber,
                    header_text=" ",
                    body_text=body_text,
                    footer_text="Powered by AI",
                    buttons=buttons,
                    conversation_id=None,
                    access_token=request.accessToken
                )
            except Exception as wa_error:
                print(f"Warning: Failed to send WhatsApp message: {str(wa_error)}")
                # Don't fail the request if WhatsApp sending fails

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

        # Process PDF using common function
        result = await process_pdf_to_vectors(
            file_content=file_content,
            filename=file.filename,
            entity=entity,
            url=None  # File upload, not URL
        )

        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "Failed to process PDF")
            )

        # Log the request to Qdrant
        try:
            session_id = str(uuid.uuid4())
            user_doc = auth_data.get("user", {})

            request_log_service.log_request(
                entity=entity,
                session=session_id,
                operation_type="pdf_upload",
                user_id=auth_data.get("user_id"),
                user_name=user_doc.get("name", ""),
                user_mobile=user_doc.get("mobile", ""),
                pdf_url=None,  # File upload, not URL
                website_url=None,
                metadata={
                    "filename": result["filename"],
                    "pages_count": result["pages_count"],
                    "chunks_created": result["chunks_created"],
                    "vectors_stored": result["vectors_stored"],
                    "status": "success"
                }
            )
        except Exception as log_error:
            print(f"Warning: Failed to log request: {str(log_error)}")
            # Don't fail the request if logging fails

        return PDFUploadResponse(
            success=True,
            message=f"Successfully processed PDF and stored {result['vectors_stored']} vectors",
            entity=entity,
            filename=result["filename"],
            pages_count=result["pages_count"],
            chunks_created=result["chunks_created"],
            vectors_stored=result["vectors_stored"],
            url=None,
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

@router.post("/uploadPdfUrl", response_model=PDFUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_pdf_url(
    request: PDFUrlRequest,
    auth_data: dict = Depends(jwt_auth.verify_token)
):
    """
    Download PDF from URL and store in vector database

    **Authentication Required**: This endpoint requires a valid JWT token in the Authorization header.

    This endpoint:
    1. Downloads PDF from provided URL
    2. Extracts text from PDF
    3. Chunks the content into sentence windows
    4. Generates embeddings for each sentence
    5. Stores in vector DB with source_type='pdf'

    - **url**: PDF URL to download
    - **entity**: Entity identifier
    """
    try:
        # Validate inputs
        if not request.entity or not request.entity.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Entity cannot be empty"
            )

        # Validate URL ends with .pdf
        url_str = str(request.url)
        if not url_str.lower().endswith('.pdf'):
            print(f"Warning: URL does not end with .pdf: {url_str}")

        print(f"Processing PDF from URL: {url_str} for entity={request.entity}")

        # Process PDF from URL using common function
        result = await process_pdf_from_url(
            url=url_str,
            entity=request.entity
        )

        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "Failed to process PDF from URL")
            )

        # Log the request to Qdrant
        try:
            session_id = str(uuid.uuid4())
            user_doc = auth_data.get("user", {})

            request_log_service.log_request(
                entity=request.entity,
                session=session_id,
                operation_type="pdf_url_upload",
                user_id=auth_data.get("user_id"),
                user_name=user_doc.get("name", ""),
                user_mobile=user_doc.get("mobile", ""),
                pdf_url=url_str,
                website_url=None,
                metadata={
                    "filename": result["filename"],
                    "pages_count": result["pages_count"],
                    "chunks_created": result["chunks_created"],
                    "vectors_stored": result["vectors_stored"],
                    "status": "success"
                }
            )
        except Exception as log_error:
            print(f"Warning: Failed to log request: {str(log_error)}")
            # Don't fail the request if logging fails

        return PDFUploadResponse(
            success=True,
            message=f"Successfully processed PDF from URL and stored {result['vectors_stored']} vectors",
            entity=request.entity,
            filename=result["filename"],
            pages_count=result["pages_count"],
            chunks_created=result["chunks_created"],
            vectors_stored=result["vectors_stored"],
            url=url_str,
            timestamp=datetime.utcnow().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in PDF URL upload endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing PDF from URL: {str(e)}"
        )

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service="leads"
    )
