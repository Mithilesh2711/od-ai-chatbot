from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import os

# Import configuration
from config.settings import API_TITLE, API_VERSION, OPENAI_API_KEY
from config.database import mongo_client

# Set OpenAI API key in environment for libraries that expect it there
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Import service routers
from services.erpService import router as erp_router
from services.vectorService import router as vector_router
from services.scrapingService import router as scraping_router
from services.leadsService import router as leads_router
from services.mtalkzService import router as mtalkz_router
from services.requestLogRoutes import router as logs_router

# FastAPI app initialization
app = FastAPI(title=API_TITLE, version=API_VERSION)

# CORS middleware - Allow all origins with no restrictions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Must be False when using wildcard origins
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],  # Expose all headers to the client
)

# Register service routers
app.include_router(erp_router)
app.include_router(vector_router)
app.include_router(scraping_router)
app.include_router(leads_router)
app.include_router(mtalkz_router)
app.include_router(logs_router)

# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint - API status check.
    """
    return {
        "status": "online",
        "service": API_TITLE,
        "version": API_VERSION
    }

# Health check endpoint
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