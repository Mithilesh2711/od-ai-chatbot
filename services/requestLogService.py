"""
Request Log Service - Store and retrieve request logs in Qdrant
Uses dummy vectors (1-dimensional) with payload indexing for fast retrieval
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, Range,
    PayloadSchemaType
)
from config.settings import QDRANT_URL, QDRANT_API_KEY
import uuid


class RequestLogService:
    """
    Service for logging requests to Qdrant with payload indexing
    Uses 1-dimensional dummy vector [0.0] since we only need relational data
    """

    def __init__(self):
        """Initialize Qdrant client and collection"""
        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )
        self.collection_name = "od-request-logs"
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist with payload indexes"""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]

            if self.collection_name not in collection_names:
                print(f"Creating Qdrant collection: {self.collection_name}")

                # Create collection with 1-dimensional vector (dummy)
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=1,  # Dummy 1-dimensional vector
                        distance=Distance.COSINE
                    )
                )

                # Create payload indexes for fast filtering
                # Index: entity
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="entity",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                print(f"✓ Created index on 'entity'")

                # Index: session
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="session",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                print(f"✓ Created index on 'session'")

                # Index: userId
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="userId",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                print(f"✓ Created index on 'userId'")

                # Index: timestamp (for range queries)
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="timestamp",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                print(f"✓ Created index on 'timestamp'")

                # Index: operationType (for filtering by operation)
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="operationType",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                print(f"✓ Created index on 'operationType'")

                # Composite indexes are handled via filtering on indexed fields
                print(f"✓ Collection '{self.collection_name}' created with indexes")

            else:
                print(f"Collection '{self.collection_name}' already exists")

        except Exception as e:
            print(f"Error ensuring collection exists: {str(e)}")
            raise

    def log_request(
        self,
        entity: str,
        session: str,
        operation_type: str,  # 'pdf_upload', 'pdf_url_upload', 'web_scraping'
        user_id: Optional[str] = None,
        user_name: Optional[str] = None,
        user_mobile: Optional[str] = None,
        pdf_url: Optional[str] = None,
        website_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a request to Qdrant

        Args:
            entity: Entity identifier
            session: Session identifier
            operation_type: Type of operation (pdf_upload, pdf_url_upload, web_scraping)
            user_id: User ID
            user_name: User name
            user_mobile: User mobile number
            pdf_url: PDF URL (if applicable)
            website_url: Website URL (if applicable)
            metadata: Additional metadata (e.g., file size, pages count, status)

        Returns:
            str: Log ID (UUID)
        """
        try:
            # Generate unique log ID
            log_id = str(uuid.uuid4())
            timestamp = datetime.utcnow().isoformat()

            # Prepare payload
            payload = {
                "entity": entity,
                "session": session,
                "operationType": operation_type,
                "userId": user_id or "",
                "userName": user_name or "",
                "userMobile": user_mobile or "",
                "pdfUrl": pdf_url or "",
                "websiteUrl": website_url or "",
                "timestamp": timestamp,
                "metadata": metadata or {}
            }

            # Create point with dummy vector [0.0]
            point = PointStruct(
                id=log_id,
                vector=[0.0],  # Dummy 1-dimensional vector
                payload=payload
            )

            # Store in Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )

            print(f"✓ Logged request: {operation_type} for entity={entity}, session={session}")
            return log_id

        except Exception as e:
            print(f"❌ Error logging request: {str(e)}")
            raise

    def update_log(
        self,
        log_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        status: Optional[str] = None
    ) -> bool:
        """
        Update an existing log entry with new metadata and/or status

        Args:
            log_id: The UUID of the log to update
            metadata: Updated metadata (will merge with existing)
            status: Updated status to set in metadata

        Returns:
            bool: True if successful
        """
        try:
            # Get existing point
            existing_points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[log_id],
                with_payload=True,
                with_vectors=False
            )

            if not existing_points:
                print(f"⚠️ Log {log_id} not found for update")
                return False

            # Get existing payload
            existing_payload = existing_points[0].payload

            # Merge metadata if provided
            if metadata:
                existing_metadata = existing_payload.get("metadata", {})
                existing_metadata.update(metadata)
                if status:
                    existing_metadata["status"] = status
                existing_payload["metadata"] = existing_metadata
            elif status:
                # Update just the status in metadata
                existing_metadata = existing_payload.get("metadata", {})
                existing_metadata["status"] = status
                existing_payload["metadata"] = existing_metadata

            # Update timestamp
            existing_payload["lastUpdated"] = datetime.utcnow().isoformat()

            # Create updated point
            point = PointStruct(
                id=log_id,
                vector=[0.0],  # Dummy vector
                payload=existing_payload
            )

            # Upsert (update) in Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )

            print(f"✓ Updated log {log_id} with status: {status}")
            return True

        except Exception as e:
            print(f"❌ Error updating log: {str(e)}")
            raise

    def get_logs(
        self,
        entity: Optional[str] = None,
        session: Optional[str] = None,
        user_id: Optional[str] = None,
        operation_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve logs with filters

        Args:
            entity: Filter by entity
            session: Filter by session
            user_id: Filter by user ID
            operation_type: Filter by operation type
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of log records
        """
        try:
            # Build filter conditions
            must_conditions = []

            if entity:
                must_conditions.append(
                    FieldCondition(
                        key="entity",
                        match=MatchValue(value=entity)
                    )
                )

            if session:
                must_conditions.append(
                    FieldCondition(
                        key="session",
                        match=MatchValue(value=session)
                    )
                )

            if user_id:
                must_conditions.append(
                    FieldCondition(
                        key="userId",
                        match=MatchValue(value=user_id)
                    )
                )

            if operation_type:
                must_conditions.append(
                    FieldCondition(
                        key="operationType",
                        match=MatchValue(value=operation_type)
                    )
                )

            # Create filter
            query_filter = None
            if must_conditions:
                query_filter = Filter(must=must_conditions)

            # Scroll through results (more efficient than search for exact matches)
            result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False  # Don't return dummy vectors
            )

            # Extract records
            records = []
            for point in result[0]:  # result is tuple (points, next_offset)
                record = {
                    "logId": point.id,
                    **point.payload
                }
                records.append(record)

            print(f"✓ Retrieved {len(records)} log records")
            return records

        except Exception as e:
            print(f"❌ Error retrieving logs: {str(e)}")
            raise

    def get_logs_by_date_range(
        self,
        entity: str,
        session: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve logs by date range (for entity_session_timestamp index use case)

        Args:
            entity: Entity identifier (required)
            session: Session identifier (optional)
            start_date: Start date in ISO format (optional)
            end_date: End date in ISO format (optional)
            limit: Maximum number of results

        Returns:
            List of log records
        """
        try:
            must_conditions = [
                FieldCondition(
                    key="entity",
                    match=MatchValue(value=entity)
                )
            ]

            if session:
                must_conditions.append(
                    FieldCondition(
                        key="session",
                        match=MatchValue(value=session)
                    )
                )

            # Note: Range filtering on timestamp as string (ISO format)
            # For better date range queries, consider storing timestamp as integer (epoch)
            if start_date or end_date:
                range_condition = {}
                if start_date:
                    range_condition["gte"] = start_date
                if end_date:
                    range_condition["lte"] = end_date

                must_conditions.append(
                    FieldCondition(
                        key="timestamp",
                        range=Range(**range_condition)
                    )
                )

            query_filter = Filter(must=must_conditions)

            result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )

            records = []
            for point in result[0]:
                record = {
                    "logId": point.id,
                    **point.payload
                }
                records.append(record)

            print(f"✓ Retrieved {len(records)} log records by date range")
            return records

        except Exception as e:
            print(f"❌ Error retrieving logs by date range: {str(e)}")
            raise

    def count_logs(
        self,
        entity: Optional[str] = None,
        session: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> int:
        """
        Count logs matching filters

        Args:
            entity: Filter by entity
            session: Filter by session
            user_id: Filter by user ID

        Returns:
            int: Count of matching logs
        """
        try:
            must_conditions = []

            if entity:
                must_conditions.append(
                    FieldCondition(key="entity", match=MatchValue(value=entity))
                )
            if session:
                must_conditions.append(
                    FieldCondition(key="session", match=MatchValue(value=session))
                )
            if user_id:
                must_conditions.append(
                    FieldCondition(key="userId", match=MatchValue(value=user_id))
                )

            query_filter = Filter(must=must_conditions) if must_conditions else None

            # Use count API
            count_result = self.client.count(
                collection_name=self.collection_name,
                count_filter=query_filter,
                exact=True
            )

            return count_result.count

        except Exception as e:
            print(f"❌ Error counting logs: {str(e)}")
            raise


# Global instance
request_log_service = RequestLogService()
