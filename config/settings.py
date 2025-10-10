import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# MongoDB Configuration
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://app:jfudDUFIEPsddf4329KFNnn@qadb.odpay.in:27023/okiedokieERP?replicaSet=mdbqars0&authSource=admin")
DATABASE_NAME = os.getenv("DATABASE_NAME", "okiedokieERP")

# OpenAI Configuration
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "https://3bd62d09-216e-40df-90e4-d75bdcb97389.eu-central-1-0.aws.cloud.qdrant.io:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.4llmwBuD0LqHIxnrni4OEpoPeh1eji4vQWVpKWTcoXk")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "od-leads")

# Embedding Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")  # OpenAI embedding model
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1536"))  # OpenAI dimension

# Tool Configuration
TOOL_BASE_URL = os.getenv("TOOL_BASE_URL", "http://localhost:7070")

# API Configuration
API_TITLE = "AI Chatbot API"
API_VERSION = "1.0.0"
