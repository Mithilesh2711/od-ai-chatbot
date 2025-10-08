import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# MongoDB Configuration
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://app:jfudDUFIEPsddf4329KFNnn@qadb.odpay.in:27023/okiedokieERP?replicaSet=mdbqars0&authSource=admin")
DATABASE_NAME = os.getenv("DATABASE_NAME", "okiedokieERP")

# OpenAI Configuration
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Tool Configuration
TOOL_BASE_URL = os.getenv("TOOL_BASE_URL", "http://localhost:7070")

# API Configuration
API_TITLE = "AI Chatbot API"
API_VERSION = "1.0.0"
