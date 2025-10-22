from typing import Dict, Any, Optional
from fastapi import HTTPException
from bson import ObjectId
from config.database import students_collection, users_collection, communication_configs_collection, entities_collection
from utils.helpers import mongo_to_json, normalize_phone_number

def get_student_by_phone(phone: str) -> Optional[Dict[str, Any]]:
    """
    Get student by phone number.
    Returns student data in JSON-serializable format or None if not found.
    READ ONLY - No modifications to database.
    """
    normalized_phone = normalize_phone_number(phone)
    student = students_collection.find_one({"phone": normalized_phone})

    if student:
        return mongo_to_json(student)

    return None

def get_user_by_phone(phone: str) -> Optional[Dict[str, Any]]:
    """
    Get user by phone number.
    Returns user data in JSON-serializable format or None if not found.
    READ ONLY - No modifications to database.
    """
    normalized_phone = normalize_phone_number(phone)
    user = users_collection.find_one({"phone": normalized_phone})

    if user:
        return mongo_to_json(user)

    return None

def validate_user(from_number: str) -> Optional[Dict[str, Any]]:
    """
    Validates user by checking students table first, then users table.
    Returns user data with userType and entity.
    READ ONLY - No modifications to database.
    """
    # Normalize phone number for database lookup (remove 91 if present)
    normalized_phone = normalize_phone_number(from_number)

    # Check students table first
    student_json = get_student_by_phone(normalized_phone)
    if student_json:
        return {
            "userType": "student",  # Found in students table
            "entity": student_json.get("entity", "student"),  # Get entity from student data or default to "student"
            "phone": normalized_phone,  # Use only normalized phone throughout
            "session": student_json.get("session"),
            "data": student_json
        }

    # Check users table if not found in students
    user_json = get_user_by_phone(normalized_phone)
    if user_json:
        return {
            "userType": "user",  # Found in users table
            "entity": user_json.get("entity", "user"),  # Get entity from user data or default to "user"
            "phone": normalized_phone,  # Use only normalized phone throughout
            "session": None,
            "data": user_json
        }

    # Not found in either table
    return None

def validate_communication_config(to_number: str) -> Dict[str, Any]:
    """
    Validates communication configuration for the given toNumber.
    Returns the matching chatBotConfig or raises error if not found.
    READ ONLY - No modifications to database.
    """
    # Find config document with matching senderPhone in chatBotConfig
    config_doc = communication_configs_collection.find_one({
        "chatBotConfig.senderPhone": to_number
    })

    if not config_doc:
        raise HTTPException(
            status_code=404,
            detail=f"No communication config document found for toNumber {to_number}"
        )

    # Convert to JSON-serializable format
    config_doc_json = mongo_to_json(config_doc)

    # Find specific chatBotConfig matching toNumber
    chat_bot_configs = config_doc_json.get("chatBotConfig", [])
    chat_bot_config = None

    for config in chat_bot_configs:
        if config.get("senderPhone") == to_number:
            chat_bot_config = config
            break

    if not chat_bot_config:
        raise HTTPException(
            status_code=404,
            detail=f"chatBotConfig not found for senderPhone {to_number} in communication config document"
        )

    return chat_bot_config

def get_entity_name(entity_id: str) -> str:
    """
    Get entity name by entity _id.
    Returns entity name or the entity_id if not found.
    READ ONLY - No modifications to database.
    """
    try:
        # Convert string to ObjectId
        object_id = ObjectId(entity_id)

        # Find entity by _id
        entity = entities_collection.find_one({"_id": object_id})

        if entity:
            # Return entity name (check common field names)
            entity_name = entity.get("name") or entity.get("entityName") or entity.get("title")
            if entity_name:
                return entity_name

        # If not found or no name field, return the entity_id itself
        print(f"Warning: Entity name not found for entity_id: {entity_id}")
        return entity_id

    except Exception as e:
        print(f"Error fetching entity name for {entity_id}: {str(e)}")
        # Return entity_id as fallback
        return entity_id
