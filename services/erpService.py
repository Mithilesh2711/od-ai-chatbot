from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from datetime import datetime

# Import from new structure
from db.erpDb import validate_user, validate_communication_config
from graphs.erpGraph import State, build_graph
from utils.helpers import normalize_phone_number
from utils.session import get_session_history, save_to_session, clear_session, session_exists

# Create APIRouter for ERP service
router = APIRouter(prefix="/api/erpService", tags=["ERP Service"])

# Initialize the graph
graph = build_graph()

# Request/Response models
class ChatRequest(BaseModel):
    query: str
    fromNumber: str
    toNumber: str

class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: datetime
    tool_calls: Optional[List[Dict[str, Any]]] = None

# API Endpoints
@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint that processes messages through LangGraph.
    Validates user and communication config before processing.
    """
    try:
        # Step 1: Validate user (check students first, then users)
        user_data = validate_user(request.fromNumber)
        if not user_data:
            raise HTTPException(
                status_code=404,
                detail=f"User with phone number {request.fromNumber} not found in students or users table"
            )

        # Step 2: Validate communication configuration (will raise HTTPException if not found)
        chat_config = validate_communication_config(request.toNumber)

        # Generate session ID based on fromNumber and toNumber
        session_id = f"{request.fromNumber}_{request.toNumber}_{datetime.utcnow().strftime('%Y%m%d')}"

        # Get conversation history from session storage
        history_messages = []
        session_history = get_session_history(session_id, limit=10)
        for conv in session_history:
            history_messages.append(HumanMessage(content=conv["user_message"]))
            history_messages.append(AIMessage(content=conv["bot_response"]))

        # Add current query to messages
        history_messages.append(HumanMessage(content=request.query))

        # Prepare initial state with all required data
        # Use normalized phone numbers throughout
        initial_state = State(
            messages=history_messages,
            session_id=session_id,
            user_data=user_data,
            chat_config=chat_config,
            from_number=normalize_phone_number(request.fromNumber),  # Use normalized phone
            to_number=request.toNumber  # toNumber is already in correct format
        )

        # Invoke the graph
        result = await graph.ainvoke(initial_state)

        # Extract response - could be AIMessage or last message if tools were called
        last_message = result['messages'][-1]
        print(last_message)
        if isinstance(last_message, ToolMessage):
            # If last message is tool result, get the AI message before it
            for msg in reversed(result['messages']):
                if isinstance(msg, AIMessage):
                    bot_response = msg.content
                    break
            else:
                bot_response = "I processed your request but couldn't generate a response."
        else:
            bot_response = last_message.content if hasattr(last_message, 'content') else str(last_message)

        tool_calls = result.get('tool_results')

        # Store conversation in session
        save_to_session(session_id, request.query, bot_response, tool_calls)

        # Return response
        return ChatResponse(
            response=bot_response,
            session_id=session_id,
            timestamp=datetime.utcnow(),
            tool_calls=tool_calls
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/conversations/{session_id}")
async def get_conversation(session_id: str, limit: int = 10):
    """
    Retrieves conversation history for a given session from session storage.
    """
    try:
        conversations = get_session_history(session_id, limit)
        return {
            "session_id": session_id,
            "conversations": conversations,
            "count": len(conversations)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/conversations/{session_id}")
async def clear_conversation(session_id: str):
    """
    Clears conversation history from session storage.
    """
    try:
        count = clear_session(session_id)
        if count > 0:
            return {
                "session_id": session_id,
                "cleared_count": count,
                "message": "Cleared from memory only (database is read-only)"
            }
        return {
            "session_id": session_id,
            "cleared_count": 0,
            "message": "No conversation found in memory"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
