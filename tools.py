from typing import Dict, Any
from langchain.tools import tool
import httpx
import os

# Base URL for tool endpoints
TOOL_BASE_URL = os.getenv("TOOL_BASE_URL", "http://localhost:7070")

# Helper function to make tool API calls
async def make_tool_call(action_id: str, user_data: Dict, from_number: str, to_number: str) -> Dict[str, Any]:
    """Makes API call to tool endpoint with standard payload"""
    endpoint = f"{TOOL_BASE_URL}/api/{action_id}/chatBotTool"

    payload = {
        "actionId": action_id,
        "entity": user_data.get("entity", "unknown"),  # Will come from student.entity or user.entity
        "phone": from_number,
        "userType": user_data.get("userType", "unknown"),  # "student" if found in students table, "user" if in users table
        "fromNumber": from_number,
        "toNumber": to_number,
        "session": user_data.get("session", None)  # Only available for students
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

# Fee Related Tools
@tool
async def userFeeToday() -> Dict[str, Any]:
    """Get Today's Fee Collection - Retrieves fee collection summary for today including total amount collected, number of transactions, and breakdown by courses"""
    return {"pending": "Tool will be called with context"}

@tool
async def userFeeMonthly() -> Dict[str, Any]:
    """Get Monthly Fee Collection Summary - Retrieves month-wise fee collection data for the current academic session"""
    return {"pending": "Tool will be called with context"}

@tool
async def userFeeDuesCourse() -> Dict[str, Any]:
    """Get Course-wise Fee Dues - Retrieves pending fee dues organized by course/class"""
    return {"pending": "Tool will be called with context"}

@tool
async def studFeeDues() -> Dict[str, Any]:
    """Get Individual Student Fee Dues - Retrieves fee dues for a specific student based on phone number"""
    return {"pending": "Tool will be called with context"}

@tool
async def studFeeTxnCurrSess() -> Dict[str, Any]:
    """Get Student Payment History (Current Session) - Retrieves payment transaction history for a student in the current academic session"""
    return {"pending": "Tool will be called with context"}

@tool
async def userFeeHeadToday() -> Dict[str, Any]:
    """Get Today's Fee Head Collection - Retrieves today's fee collection breakdown by fee heads (tuition, transport, etc.)"""
    return {"pending": "Tool will be called with context"}

@tool
async def userFeeHead7d() -> Dict[str, Any]:
    """Get Last 7 Days Fee Head Collection - Retrieves fee collection by heads for the last 7 days"""
    return {"pending": "Tool will be called with context"}

@tool
async def userFeeHeadMonth() -> Dict[str, Any]:
    """Get Current Month Fee Head Collection - Retrieves fee collection by heads for the current month"""
    return {"pending": "Tool will be called with context"}

@tool
async def userFeeHead30d() -> Dict[str, Any]:
    """Get Last 30 Days Fee Head Collection - Retrieves fee collection by heads for the last 30 days"""
    return {"pending": "Tool will be called with context"}

@tool
async def userPayModeToday() -> Dict[str, Any]:
    """Get Today's Payment Mode Collection - Retrieves today's fee collection breakdown by payment modes (cash, online, cheque, etc.)"""
    return {"pending": "Tool will be called with context"}

@tool
async def userPayMode7d() -> Dict[str, Any]:
    """Get Last 7 Days Payment Mode Collection - Retrieves payment mode wise collection for the last 7 days"""
    return {"pending": "Tool will be called with context"}

@tool
async def userPayModeMonth() -> Dict[str, Any]:
    """Get Current Month Payment Mode Collection - Retrieves payment mode wise collection for the current month"""
    return {"pending": "Tool will be called with context"}

@tool
async def userPayMode30d() -> Dict[str, Any]:
    """Get Last 30 Days Payment Mode Collection - Retrieves payment mode wise collection for the last 30 days"""
    return {"pending": "Tool will be called with context"}

@tool
async def userFeeConcGroup() -> Dict[str, Any]:
    """Get Concession Group Waiver Report - Retrieves fee waiver/concession data grouped by concession categories"""
    return {"pending": "Tool will be called with context"}

# Attendance Related Tools
@tool
async def studAttToday() -> Dict[str, Any]:
    """Get Student Today's Attendance - Retrieves attendance status for a specific student for today"""
    return {"pending": "Tool will be called with context"}

@tool
async def studAttWeek() -> Dict[str, Any]:
    """Get Student Weekly Attendance Summary - Retrieves attendance summary for a student for the current week"""
    return {"pending": "Tool will be called with context"}

@tool
async def studAttMonth() -> Dict[str, Any]:
    """Get Student Monthly Attendance Summary - Retrieves attendance summary for a student for the current month"""
    return {"pending": "Tool will be called with context"}

@tool
async def studAttReport() -> Dict[str, Any]:
    """Get Student Attendance Report - Generates comprehensive attendance report for a student for specified period"""
    return {"pending": "Tool will be called with context"}

@tool
async def userAttTodayAll() -> Dict[str, Any]:
    """Get Today's Overall Attendance - Retrieves overall attendance statistics for all students today"""
    return {"pending": "Tool will be called with context"}

# Admission Related Tools
@tool
async def userAdmOverall() -> Dict[str, Any]:
    """Get Overall Admission Summary - Retrieves overall admission statistics for the current academic session"""
    return {"pending": "Tool will be called with context"}

@tool
async def userAdmCourse() -> Dict[str, Any]:
    """Get Course-wise Admission Summary - Retrieves admission statistics broken down by course/class"""
    return {"pending": "Tool will be called with context"}

@tool
async def userAdmCategory() -> Dict[str, Any]:
    """Get Category-wise Admission Summary - Retrieves admission statistics broken down by student categories (General, SC/ST, etc.)"""
    return {"pending": "Tool will be called with context"}

# Tool categorization by user type
STUDENT_TOOLS = [
    # Student-specific fee tools
    studFeeDues,
    studFeeTxnCurrSess,
    # Student-specific attendance tools
    studAttToday,
    studAttWeek,
    studAttMonth,
    studAttReport
]

USER_TOOLS = [
    # User/Admin fee tools
    userFeeToday,
    userFeeMonthly,
    userFeeDuesCourse,
    userFeeHeadToday,
    userFeeHead7d,
    userFeeHeadMonth,
    userFeeHead30d,
    userPayModeToday,
    userPayMode7d,
    userPayModeMonth,
    userPayMode30d,
    userFeeConcGroup,
    # User/Admin attendance tools
    userAttTodayAll,
    # User/Admin admission tools
    userAdmOverall,
    userAdmCourse,
    userAdmCategory
]

# Export all tools as a list
def get_all_tools():
    """Returns list of all available tools"""
    return STUDENT_TOOLS + USER_TOOLS

# Export tools filtered by user type
def get_tools_for_user_type(user_type: str):
    """Returns tools filtered by user type (student or user)"""
    if user_type == "student":
        return STUDENT_TOOLS
    elif user_type == "user":
        return USER_TOOLS
    else:
        return []  # No tools for unknown user types