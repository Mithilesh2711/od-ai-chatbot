from typing import List, Dict, Any
from datetime import datetime

# Session management - In-memory storage (no DB writes)
# Using in-memory storage since we cannot write to the database
conversation_sessions = {}

def get_session_history(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get conversation history for a given session.
    Returns last 'limit' conversations.
    """
    if session_id not in conversation_sessions:
        return []

    return conversation_sessions[session_id][-limit:]

def save_to_session(session_id: str, user_message: str, bot_response: str, tool_calls: Any = None):
    """
    Save a conversation to session.
    Maintains only last 50 conversations per session.
    """
    if session_id not in conversation_sessions:
        conversation_sessions[session_id] = []

    conversation_sessions[session_id].append({
        "user_message": user_message,
        "bot_response": bot_response,
        "timestamp": datetime.utcnow(),
        "tool_calls": tool_calls
    })

    # Keep only last 50 conversations per session in memory
    if len(conversation_sessions[session_id]) > 50:
        conversation_sessions[session_id] = conversation_sessions[session_id][-50:]

def clear_session(session_id: str) -> int:
    """
    Clear conversation history for a session.
    Returns count of cleared conversations.
    """
    if session_id in conversation_sessions:
        count = len(conversation_sessions[session_id])
        del conversation_sessions[session_id]
        return count
    return 0

def session_exists(session_id: str) -> bool:
    """Check if session exists"""
    return session_id in conversation_sessions
