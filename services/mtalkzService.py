from fastapi import APIRouter, HTTPException, status, Request
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
import httpx

# Import database functions
from db.erpDb import validate_user, validate_communication_config

# Import leads service function
from services.leadsService import process_chat

# Import configuration
from config.settings import MTALKZ_BASE_URL, MTALKZ_API_KEY, MTALKZ_ACCESS_TOKEN

router = APIRouter(prefix="/api/mtalkz", tags=["mTalkz"])

# Request/Response Models
class MessageContent(BaseModel):
    """WhatsApp message content from mTalkz webhook"""
    from_: str = Field(..., alias="from", description="Sender phone number")
    to: str = Field(..., description="Recipient phone number (bot)")
    contentType: str = Field(..., description="Message content type (text, interactive, etc.)")
    text: Optional[Dict[str, Any]] = None
    interactive: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None

class EventContent(BaseModel):
    """Event content wrapper"""
    message: MessageContent

class WebhookRequest(BaseModel):
    """mTalkz webhook request body"""
    eventContent: EventContent

class WebhookResponse(BaseModel):
    """Webhook response"""
    success: bool
    message: str
    entity: Optional[str] = None
    userType: Optional[str] = None
    response: Optional[str] = None

class ChatTestRequest(BaseModel):
    """Test API request model"""
    phone: str = Field(..., description="User phone number")
    query: str = Field(..., description="User query")

class ChatTestResponse(BaseModel):
    """Test API response model"""
    success: bool
    entity: str
    userType: str
    query: str
    answer: str
    phone: str

# Helper function to send interactive button message
async def send_interactive_button_message(
    to_phone: str,
    from_phone: str,
    header_text: str,
    body_text: str,
    footer_text: str,
    buttons: List[Dict[str, Any]],
    conversation_id: Optional[str] = None,
    access_token: Optional[str] = None
) -> Dict[str, Any]:
    """
    Send WhatsApp interactive button message via mTalkz API.

    Args:
        to_phone: Recipient phone number
        from_phone: Sender phone number
        header_text: Header text for the message
        body_text: Main body text (AI response)
        footer_text: Footer text
        buttons: List of button dicts with 'id' and 'title' keys (max 3 buttons)
        conversation_id: Optional conversation ID
        access_token: mTalkz access token

    Returns:
        API response dict
    """
    # Use access token from settings if not provided
    token = access_token or MTALKZ_ACCESS_TOKEN

    # Format buttons for mTalkz API (max 3 buttons)
    formatted_buttons = []
    for btn in buttons[:3]:  # WhatsApp allows max 3 buttons
        formatted_buttons.append({
            "type": "reply",
            "reply": {
                "id": btn.get("id", ""),
                "title": btn.get("title", "")[:20]  # Max 20 chars for button title
            }
        })

    payload = {
        "message": {
            "channel": "WABA",
            "content": {
                "preview_url": False,
                "shorten_url": False,
                "type": "INTERACTIVE",
                "interactive": {
                    "type": "button",
                    "header": {
                        "type": "text",
                        "text": header_text
                    },
                    "body": {
                        "text": body_text
                    },
                    "footer": {
                        "text": footer_text
                    },
                    "action": {
                        "buttons": formatted_buttons
                    }
                }
            },
            "recipient": {
                "to": to_phone,
                "recipient_type": "individual",
                "reference": {
                    "cust_ref": to_phone,
                    "messageTag1": "",
                    "conversationId": conversation_id or f"{to_phone}-{datetime.utcnow().timestamp()}"
                }
            },
            "sender": {
                "from": from_phone
            },
            "preferences": {
                "webHookDNId": "1001"
            }
        },
        "metaData": {
            "version": "v1.0.9"
        }
    }

    headers = {
        "Authentication": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(MTALKZ_BASE_URL, json=payload, headers=headers)
            print(f"mTalkz interactive message response status: {response.status_code}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            print(f"Error sending interactive message: {str(e)}")
            return {"error": str(e)}

# Helper function to build interactive message with entity URLs
def build_interactive_message(entity: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build interactive message with entity URLs from retrieved documents.
    """
    # Extract unique URLs from retrieved docs
    urls = []
    for doc in retrieved_docs:
        url = doc.get("url")
        title = doc.get("title", "View more")
        if url and url not in [u["url"] for u in urls]:
            urls.append({"url": url, "title": title})

    # Build interactive list (max 10 items)
    sections = []
    if urls:
        rows = []
        for idx, url_data in enumerate(urls[:10]):
            rows.append({
                "id": f"url_{idx}",
                "title": url_data["title"][:24],  # WhatsApp max 24 chars
                "description": url_data["url"][:72]  # WhatsApp max 72 chars
            })

        sections.append({
            "title": "Resources",
            "rows": rows
        })

    interactive = {
        "type": "list",
        "header": {
            "type": "text",
            "text": f"Information about {entity}"
        },
        "body": {
            "text": "Here are some helpful resources:"
        },
        "footer": {
            "text": "Powered by AI"
        },
        "action": {
            "button": "View Resources",
            "sections": sections
        }
    }

    return interactive

@router.post("/webhook", response_model=WebhookResponse)
async def mtalkz_webhook(request: WebhookRequest):
    """
    mTalkz webhook endpoint to receive WhatsApp messages.

    Process flow:
    1. Receive and validate webhook payload
    2. Extract message details (from, to, content)
    3. Query ERP database to get entity by phone
    4. Call leads service chat API with entity and query
    5. Send response back to user (for now just return the response)
    """
    try:
        # Extract message details
        message = request.eventContent.message
        from_phone = message.from_
        to_phone = message.to
        content_type = message.contentType

        # Validate required fields
        if not from_phone or not to_phone or not content_type:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid webhook format: missing from, to, or contentType"
            )

        print(f"Received webhook from {from_phone} to {to_phone}, type: {content_type}")

        # Extract message text based on content type
        user_query = ""
        if content_type == "text" and message.text:
            user_query = message.text.get("body", "")
        elif content_type == "interactive" and message.interactive:
            # Handle interactive message response
            interactive_type = message.interactive.get("type", "")
            if interactive_type in ["list_reply", "button_reply"]:
                user_query = message.interactive.get(interactive_type, {}).get("title", "")

        if not user_query:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not extract message text from webhook"
            )

        # Validate communication config to get accessToken and model from chatBotConfig
        try:
            chat_bot_config = validate_communication_config(to_phone)
            access_token = chat_bot_config.get("accessToken", "")
            sender_phone = chat_bot_config.get("senderPhone", to_phone)
            model_name = chat_bot_config.get("model", None)  # Get model from chatBotConfig
            print(f"Found chatBotConfig for {to_phone}, name: {chat_bot_config.get('name', 'N/A')}, model: {model_name or 'default'}")
        except HTTPException as e:
            return WebhookResponse(
                success=False,
                message=f"Communication config not found for toNumber: {to_phone}",
                entity=None,
                userType=None,
                response=None
            )

        # Get entity from ERP database using validate_user
        user_data = validate_user(from_phone)

        if not user_data:
            return WebhookResponse(
                success=False,
                message=f"User/Student not found for phone: {from_phone}",
                entity=None,
                userType=None,
                response=None
            )

        entity = user_data["entity"]
        user_type = user_data["userType"]

        print(f"Found {user_type} with entity: {entity}")

        # Call leads service process_chat function directly with model_name
        # This will fetch entity_name internally (only once)
        chat_result = await process_chat(entity, user_query, model_name=model_name)

        answer = chat_result.get("answer", "")
        entity_name = chat_result.get("entity_name", entity)
        retrieved_docs = chat_result.get("retrieved_docs", [])

        print(f"Entity name: {entity_name}")

        # Send WhatsApp response back to user
        print(f"Chat response: {answer}")

        # Get conversation ID from context if available
        conversation_id = None
        if message.context:
            conversation_id = message.context.get("id")

        # Send interactive button message via WhatsApp
        if access_token:
            # Prepare buttons - always include Main Menu button
            buttons = [
                {
                    "id": "main_menu",
                    "title": "Main Menu"
                }
            ]

            # Truncate answer if too long (WhatsApp body max ~1024 chars)
            body_text = answer[:1000] if len(answer) > 1000 else answer

            # Send interactive message with buttons
            whatsapp_result = await send_interactive_button_message(
                to_phone=from_phone,
                from_phone=sender_phone,
                header_text=f" ",
                body_text=body_text,
                footer_text="Powered by AI",
                buttons=buttons,
                conversation_id=conversation_id,
                access_token=access_token
            )
            print(f"WhatsApp interactive message sent: {whatsapp_result}")
        else:
            print("Warning: No access token found in chatBotConfig, skipping WhatsApp response")

        return WebhookResponse(
            success=True,
            message="Webhook received and processed successfully",
            entity=entity,
            userType=user_type,
            response=answer
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing webhook: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing webhook: {str(e)}"
        )

@router.post("/chat/test", response_model=ChatTestResponse)
async def test_chat(request: ChatTestRequest):
    """
    Test endpoint to test the chat flow without webhook.
    This calls the leads service chat API directly.
    """
    try:
        # Get entity from ERP database using validate_user
        user_data = validate_user(request.phone)

        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User/Student not found for phone: {request.phone}"
            )

        entity = user_data["entity"]
        user_type = user_data["userType"]

        # Call leads service process_chat function directly
        chat_result = await process_chat(entity, request.query)

        answer = chat_result.get("answer", "No response generated")

        return ChatTestResponse(
            success=True,
            entity=entity,
            userType=user_type,
            query=request.query,
            answer=answer,
            phone=request.phone
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in test chat: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing test chat: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "mtalkz",
        "timestamp": datetime.utcnow().isoformat()
    }
