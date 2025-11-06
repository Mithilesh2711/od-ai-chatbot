"""
Request Log Routes - API endpoints for retrieving request logs
"""

from fastapi import APIRouter, HTTPException, status, Depends, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from services.requestLogService import request_log_service
from middleware.auth import jwt_auth

router = APIRouter(prefix="/api/logs", tags=["Request Logs"])


# Response Models
class LogRecord(BaseModel):
    logId: str
    entity: str
    session: str
    operationType: str
    userId: Optional[str] = ""
    userName: Optional[str] = ""
    userMobile: Optional[str] = ""
    pdfUrl: Optional[str] = ""
    websiteUrl: Optional[str] = ""
    timestamp: str
    metadata: Dict[str, Any] = {}


class LogListResponse(BaseModel):
    success: bool
    count: int
    total: int
    logs: List[LogRecord]
    filters: Dict[str, Any]


class LogCountResponse(BaseModel):
    success: bool
    count: int
    filters: Dict[str, Any]


@router.get("/list", response_model=LogListResponse)
async def list_logs(
    entity: Optional[str] = Query(None, description="Filter by entity"),
    session: Optional[str] = Query(None, description="Filter by session"),
    userId: Optional[str] = Query(None, description="Filter by user ID"),
    operationType: Optional[str] = Query(None, description="Filter by operation type (pdf_upload, pdf_url_upload, web_scraping)"),
    limit: int = Query(100, ge=1, le=500, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    auth_data: dict = Depends(jwt_auth.verify_token)
):
    """
    List request logs with filters

    **Authentication Required**: This endpoint requires a valid JWT token in the Authorization header.

    Filters:
    - **entity**: Filter by entity identifier
    - **session**: Filter by session identifier
    - **userId**: Filter by user ID
    - **operationType**: Filter by operation type (pdf_upload, pdf_url_upload, web_scraping)
    - **limit**: Maximum number of results (default: 100, max: 500)
    - **offset**: Offset for pagination (default: 0)

    Returns list of log records matching the filters.
    """
    try:
        # Get logs with filters
        logs = request_log_service.get_logs(
            entity=entity,
            session=session,
            user_id=userId,
            operation_type=operationType,
            limit=limit,
            offset=offset
        )

        # Get total count (without pagination)
        total_count = request_log_service.count_logs(
            entity=entity,
            session=session,
            user_id=userId
        )

        # Format filters for response
        applied_filters = {}
        if entity:
            applied_filters["entity"] = entity
        if session:
            applied_filters["session"] = session
        if userId:
            applied_filters["userId"] = userId
        if operationType:
            applied_filters["operationType"] = operationType
        applied_filters["limit"] = limit
        applied_filters["offset"] = offset

        # Convert to LogRecord models
        log_records = [LogRecord(**log) for log in logs]

        return LogListResponse(
            success=True,
            count=len(log_records),
            total=total_count,
            logs=log_records,
            filters=applied_filters
        )

    except Exception as e:
        print(f"Error in list_logs endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving logs: {str(e)}"
        )


@router.get("/listByDateRange", response_model=LogListResponse)
async def list_logs_by_date_range(
    entity: str = Query(..., description="Entity identifier (required)"),
    session: Optional[str] = Query(None, description="Filter by session"),
    startDate: Optional[str] = Query(None, description="Start date (ISO format: YYYY-MM-DDTHH:MM:SS)"),
    endDate: Optional[str] = Query(None, description="End date (ISO format: YYYY-MM-DDTHH:MM:SS)"),
    limit: int = Query(100, ge=1, le=500, description="Maximum number of results"),
    auth_data: dict = Depends(jwt_auth.verify_token)
):
    """
    List request logs by date range (uses entity_session_timestamp index)

    **Authentication Required**: This endpoint requires a valid JWT token in the Authorization header.

    Filters:
    - **entity**: Entity identifier (required)
    - **session**: Filter by session identifier (optional)
    - **startDate**: Start date in ISO format (e.g., "2025-01-01T00:00:00")
    - **endDate**: End date in ISO format (e.g., "2025-01-31T23:59:59")
    - **limit**: Maximum number of results (default: 100, max: 500)

    Returns list of log records within the specified date range.
    """
    try:
        # Validate entity
        if not entity or not entity.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Entity is required"
            )

        # Get logs by date range
        logs = request_log_service.get_logs_by_date_range(
            entity=entity,
            session=session,
            start_date=startDate,
            end_date=endDate,
            limit=limit
        )

        # Get total count
        total_count = request_log_service.count_logs(
            entity=entity,
            session=session
        )

        # Format filters for response
        applied_filters = {
            "entity": entity,
            "limit": limit
        }
        if session:
            applied_filters["session"] = session
        if startDate:
            applied_filters["startDate"] = startDate
        if endDate:
            applied_filters["endDate"] = endDate

        # Convert to LogRecord models
        log_records = [LogRecord(**log) for log in logs]

        return LogListResponse(
            success=True,
            count=len(log_records),
            total=total_count,
            logs=log_records,
            filters=applied_filters
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in list_logs_by_date_range endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving logs by date range: {str(e)}"
        )


@router.get("/count", response_model=LogCountResponse)
async def count_logs(
    entity: Optional[str] = Query(None, description="Filter by entity"),
    session: Optional[str] = Query(None, description="Filter by session"),
    userId: Optional[str] = Query(None, description="Filter by user ID"),
    auth_data: dict = Depends(jwt_auth.verify_token)
):
    """
    Count request logs matching filters

    **Authentication Required**: This endpoint requires a valid JWT token in the Authorization header.

    Filters:
    - **entity**: Filter by entity identifier
    - **session**: Filter by session identifier
    - **userId**: Filter by user ID

    Returns count of log records matching the filters.
    """
    try:
        # Get count
        count = request_log_service.count_logs(
            entity=entity,
            session=session,
            user_id=userId
        )

        # Format filters for response
        applied_filters = {}
        if entity:
            applied_filters["entity"] = entity
        if session:
            applied_filters["session"] = session
        if userId:
            applied_filters["userId"] = userId

        return LogCountResponse(
            success=True,
            count=count,
            filters=applied_filters
        )

    except Exception as e:
        print(f"Error in count_logs endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error counting logs: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "request-logs"}
