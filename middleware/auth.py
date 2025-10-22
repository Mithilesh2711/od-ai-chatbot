from fastapi import HTTPException, status, Header, Request
from typing import Optional
import jwt
from bson import ObjectId
from config.settings import JWT_SECRET_KEY, JWT_ALGORITHM
from config.database import users_collection


class JWTAuth:
    """
    JWT Authentication middleware for validating tokens against ERP database users
    Matches Node.js authentication logic - no Bearer prefix, token from header/body/query
    """

    @staticmethod
    async def verify_token(
        request: Request,
        authorization: Optional[str] = Header(None, alias="Authorization")
    ) -> dict:
        """
        Verify JWT token and validate against ERP users table

        Token can come from:
        - Authorization header (direct token, no Bearer prefix)
        - Request body 'token' field
        - Query parameter 'token'

        Args:
            request: FastAPI Request object
            authorization: Authorization header value (direct token, no Bearer)

        Returns:
            User data from database

        Raises:
            HTTPException: If token is invalid, expired, or user not found in ERP DB
        """
        # Get token from Authorization header, body, or query (matches Node.js logic)
        user_token = authorization

        if not user_token:
            # Try to get from request body
            try:
                body = await request.json()
                user_token = body.get('token')
            except:
                pass

        if not user_token:
            # Try to get from query params
            user_token = request.query_params.get('token')

        if not user_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed. Token missing."
            )

        # Decode and verify JWT token (no Bearer prefix, direct token)
        try:
            decoded = jwt.decode(
                user_token,
                JWT_SECRET_KEY,
                algorithms=[JWT_ALGORITHM]
            )
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError as e:
            error_message = str(e)
            if "Signature verification failed" in error_message or "Invalid" in error_message:
                error_message = "Authentication failed. Please log in again."
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=error_message
            )

        # Extract user ID from token (matches Node.js: decoded._id)
        user_id = decoded.get("_id")

        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token missing user identifier (_id)"
            )

        # Validate user exists in ERP database by _id
        try:
            # Convert string _id to MongoDB ObjectId
            try:
                object_id = ObjectId(user_id)
            except Exception:
                # If conversion fails, user_id might already be ObjectId or invalid
                object_id = user_id

            user_doc = users_collection.find_one({"_id": object_id})

            if not user_doc:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Session Expired, Please Login again"
                )

            print(f"Auth - User found in DB: {user_doc.get('name')}")

            # Return user data for use in endpoint
            return {
                "user_id": str(user_id),
                "user_type": decoded.get("userType"),
                "decoded": decoded,
                "user": user_doc
            }

        except HTTPException:
            raise
        except Exception as e:
            # Log the error for debugging
            print(f"Error validating user in ERP database: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error: {str(e)}"
            )


# Create global instance for dependency injection
jwt_auth = JWTAuth()
