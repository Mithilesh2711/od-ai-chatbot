from typing import Dict, Any
import httpx
from config.settings import TOOL_BASE_URL

async def make_tool_call(action_id: str, user_data: Dict, from_number: str, to_number: str, chat_config: Dict = None) -> Dict[str, Any]:
    """Makes API call to tool endpoint with standard payload"""
    endpoint = f"{TOOL_BASE_URL}/api/{action_id}/chatBotTool"

    # Ensure userType is always passed correctly
    user_type = user_data.get("userType")
    if not user_type:
        # Fallback: determine from entity if userType is missing
        if "session" in user_data:
            user_type = "student"
        else:
            user_type = "user"

    payload = {
        "actionId": action_id,
        "entity": user_data.get("entity", "unknown"),  # Will come from student.entity or user.entity
        "phone": from_number,  # Use normalized phone number
        "userType": user_type,  # Ensure this is always passed
        "fromNumber": from_number,  # Normalized phone number
        "toNumber": to_number,
        "session": user_data.get("session", None),  # Only available for students
        "chatBotConfig": chat_config  # Pass the entire chatBotConfig
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(endpoint, json=payload)
            response.raise_for_status()
            return {"status": "success", "data": response.json()}
    except httpx.HTTPStatusError as e:
        return {"status": "error", "message": str(e), "status_code": e.response.status_code}
    except Exception as e:
        return {"status": "error", "message": str(e)}
