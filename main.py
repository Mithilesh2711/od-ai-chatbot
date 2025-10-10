from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

# Import configuration
from config.settings import API_TITLE, API_VERSION
from config.database import mongo_client

# Import service routers
from services.erpService import router as erp_router
from services.vectorService import router as vector_router
from services.scrapingService import router as scraping_router
from services.leadsService import router as leads_router

# FastAPI app initialization
app = FastAPI(title=API_TITLE, version=API_VERSION)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register service routers
app.include_router(erp_router)
app.include_router(vector_router)
app.include_router(scraping_router)
app.include_router(leads_router)

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