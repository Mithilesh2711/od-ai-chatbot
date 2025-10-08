from typing import Any
from bson import ObjectId
from datetime import datetime

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

def normalize_phone_number(phone: str) -> str:
    """
    Normalize phone number by removing country code (91) if present.
    Used throughout the system for consistency.
    """
    if not phone:
        return phone

    # Remove any non-digit characters
    phone = ''.join(filter(str.isdigit, phone))

    # Remove 91 prefix if present and number is longer than 10 digits
    if phone.startswith('91') and len(phone) > 10:
        return phone[2:]

    return phone
