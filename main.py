from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv
import os
from datetime import datetime
import json

# Import graph and state from graph module
from graph import State, build_graph

# Helper function to convert MongoDB documents to JSON-serializable format
def mongo_to_json(data: Any) -> Any:
    """
    Recursively convert MongoDB documents to JSON-serializable format.
    Handles ObjectId, datetime, and other non-serializable types.
    """
    if isinstance(data, dict):
        return {key: mongo_to_json(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [mongo_to_json(item) for item in data]
    elif isinstance(data, ObjectId):
        return str(data)
    elif isinstance(data, datetime):
        return data.isoformat()
    elif hasattr(data, '__dict__'):
        return mongo_to_json(data.__dict__)
    else:
        return data

# Load environment variables
load_dotenv(override=True)

# FastAPI app initialization
app = FastAPI(title="AI Chatbot API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB configuration
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://app:jfudDUFIEPsddf4329KFNnn@qadb.odpay.in:27023/okiedokieERP?replicaSet=mdbqars0&authSource=admin")
DATABASE_NAME = os.getenv("DATABASE_NAME", "okiedokieERP")

# MongoDB client - READ ONLY DATABASE
# WARNING: This database is for READ operations only. Do NOT write, update, or delete any records.
mongo_client = MongoClient(MONGODB_URL)
db = mongo_client[DATABASE_NAME]
# Collections for READ ONLY access
students_collection = db["students"]  # READ ONLY
users_collection = db["users"]  # READ ONLY
communication_configs_collection = db["communicationconfigs"]  # READ ONLY

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


# Initialize the graph
graph = build_graph()

# Validation functions - READ ONLY operations
def validate_user(from_number: str) -> Dict[str, Any]:
    """
    Validates user by checking students table first, then users table.
    Returns user data with userType and entity.
    READ ONLY - No modifications to database.
    """
    # Check students table first
    student = students_collection.find_one({"phone": from_number})
    if student:
        # Convert to JSON-serializable format
        student_json = mongo_to_json(student)
        return {
            "userType": "student",  # Found in students table
            "entity": student_json.get("entity", "student"),  # Get entity from student data or default to "student"
            "phone": from_number,
            "session": student_json.get("session"),
            "data": student_json
        }

    # Check users table if not found in students
    user = users_collection.find_one({"phone": from_number})
    if user:
        # Convert to JSON-serializable format
        user_json = mongo_to_json(user)
        return {
            "userType": "user",  # Found in users table
            "entity": user_json.get("entity", "user"),  # Get entity from user data or default to "user"
            "phone": from_number,
            "session": None,
            "data": user_json
        }

    # Not found in either table
    return None

def validate_communication_config(to_number: str) -> Dict[str, Any]:
    """
    Validates communication configuration for the given toNumber.
    Returns the matching chatBotConfig or None.
    READ ONLY - No modifications to database.
    """
    # Find config document with matching senderPhone in chatBotConfig
    config_doc = communication_configs_collection.find_one({
        "chatBotConfig.senderPhone": to_number
    })

    if not config_doc:
        return None

    # Convert to JSON-serializable format
    config_doc_json = mongo_to_json(config_doc)

    # Find specific chatBotConfig matching toNumber
    chat_bot_configs = config_doc_json.get("chatBotConfig", [])
    for config in chat_bot_configs:
        if config.get("senderPhone") == to_number:
            return config

    return None

# Session management - In-memory storage (no DB writes)
# Using in-memory storage since we cannot write to the database
conversation_sessions = {}

# API Endpoints
@app.get("/")
async def root():
    """
    Root endpoint - API status check.
    """
    return {
        "status": "online",
        "service": "AI Chatbot API",
        "version": "1.0.0"
    }

@app.post("/api/chat", response_model=ChatResponse)
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

        # Step 2: Validate communication configuration
        chat_config = validate_communication_config(request.toNumber)
        if not chat_config:
            raise HTTPException(
                status_code=404,
                detail=f"Communication config not found for toNumber {request.toNumber}"
            )

        # Generate session ID based on fromNumber and toNumber
        session_id = f"{request.fromNumber}_{request.toNumber}_{datetime.utcnow().strftime('%Y%m%d')}"

        # Get conversation history from in-memory storage
        history_messages = []
        if session_id in conversation_sessions:
            for conv in conversation_sessions[session_id][-10:]:  # Keep last 10 messages
                history_messages.append(HumanMessage(content=conv["user_message"]))
                history_messages.append(AIMessage(content=conv["bot_response"]))

        # Add current query to messages
        history_messages.append(HumanMessage(content=request.query))

        # Prepare initial state with all required data
        initial_state = State(
            messages=history_messages,
            session_id=session_id,
            user_data=user_data,
            chat_config=chat_config,
            from_number=request.fromNumber,
            to_number=request.toNumber
        )

        # Invoke the graph
        result = await graph.ainvoke(initial_state)

        # Extract response - could be AIMessage or last message if tools were called
        last_message = result['messages'][-1]
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

        # Store conversation in memory (no DB writes)
        if session_id not in conversation_sessions:
            conversation_sessions[session_id] = []

        conversation_sessions[session_id].append({
            "user_message": request.query,
            "bot_response": bot_response,
            "timestamp": datetime.utcnow(),
            "tool_calls": tool_calls
        })

        # Keep only last 50 conversations per session in memory
        if len(conversation_sessions[session_id]) > 50:
            conversation_sessions[session_id] = conversation_sessions[session_id][-50:]

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

@app.get("/api/conversations/{session_id}")
async def get_conversation(session_id: str, limit: int = 10):
    """
    Retrieves conversation history for a given session from in-memory storage.
    """
    try:
        if session_id not in conversation_sessions:
            return {
                "session_id": session_id,
                "conversations": [],
                "count": 0
            }

        conversations = conversation_sessions[session_id][-limit:]
        return {
            "session_id": session_id,
            "conversations": conversations,
            "count": len(conversations)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/conversations/{session_id}")
async def clear_conversation(session_id: str):
    """
    Clears conversation history from in-memory storage (no DB operations).
    """
    try:
        if session_id in conversation_sessions:
            count = len(conversation_sessions[session_id])
            del conversation_sessions[session_id]
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

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    try:
        # Check MongoDB connection
        mongo_client.admin.command('ping')
        mongodb_status = "connected"
    except:
        mongodb_status = "disconnected"

    return {
        "status": "healthy",
        "mongodb": mongodb_status,
        "timestamp": datetime.utcnow()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)