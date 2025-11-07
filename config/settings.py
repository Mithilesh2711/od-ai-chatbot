# =============================================================================
# AI Chatbot Configuration
# All settings centralized in this file - no .env needed
# =============================================================================

# =============================================================================
# Multi-Model LLM Configuration
# =============================================================================

# OpenAI Configuration
OPENAI_API_KEY = "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"
OPENAI_MODEL = "gpt-3.5-turbo"  # Default model for quick responses

# Claude (Anthropic) Configuration
CLAUDE_API_KEY = ""  # Add your Claude API key here
CLAUDE_MODEL_SONNET = "claude-3-5-sonnet-20241022"  # Claude 3.5 Sonnet
CLAUDE_MODEL_HAIKU = "claude-3-5-haiku-20241022"  # Claude 3.5 Haiku

# Gemini (Google) Configuration
GEMINI_API_KEY = ""  # Add your Gemini API key here
GEMINI_MODEL_PRO = "gemini-1.5-pro"  # Gemini 1.5 Pro
GEMINI_MODEL_FLASH = "gemini-1.5-flash"  # Gemini 1.5 Flash

# DeepSeek Configuration
DEEPSEEK_API_KEY = ""  # Add your DeepSeek API key here
DEEPSEEK_MODEL_CHAT = "deepseek-chat"  # DeepSeek Chat model

# Model name mappings for chatBotConfig
MODEL_NAMES = {
    # OpenAI models
    "gpt-3.5-turbo": {"provider": "openai", "api_key": "OPENAI_API_KEY", "model": "gpt-3.5-turbo"},
    "gpt-4": {"provider": "openai", "api_key": "OPENAI_API_KEY", "model": "gpt-4"},
    "gpt-4-turbo": {"provider": "openai", "api_key": "OPENAI_API_KEY", "model": "gpt-4-turbo"},

    # Claude models
    "claude-sonnet": {"provider": "anthropic", "api_key": "CLAUDE_API_KEY", "model": "claude-3-5-sonnet-20241022"},
    "claude-haiku": {"provider": "anthropic", "api_key": "CLAUDE_API_KEY", "model": "claude-3-5-haiku-20241022"},

    # Gemini models
    "gemini-pro": {"provider": "google", "api_key": "GEMINI_API_KEY", "model": "gemini-1.5-pro"},
    "gemini-flash": {"provider": "google", "api_key": "GEMINI_API_KEY", "model": "gemini-1.5-flash"},

    # DeepSeek models
    "deepseek-chat": {"provider": "deepseek", "api_key": "DEEPSEEK_API_KEY", "model": "deepseek-chat"},
}

# Default model (used if no model specified in chatBotConfig)
DEFAULT_MODEL = "gpt-3.5-turbo"

# MongoDB Configuration
MONGODB_URL = "mongodb://app:jfudDUFIEPsddf4329KFNnn@164.52.213.143:27023/okiedokieERP?replicaSet=mdbqars0&authSource=admin"
DATABASE_NAME = "okiedokieERP"

# Qdrant Configuration
QDRANT_URL = "https://060a95ad-1c4c-44d5-b8f0-1f8191a673df.eu-west-2-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.lOrIWrrUYZZwa_rvgByqNg2dc9wRewr7KdUxSvZZuOM"
QDRANT_COLLECTION_NAME = "od-leads"

# Embedding Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 512  # Reduced for sentence window (faster, smaller storage)

# Tool Configuration
TOOL_BASE_URL = "http://localhost:7070"

# API Configuration
API_TITLE = "AI Chatbot API"
API_VERSION = "1.0.0"

# RAG Enhancement Configuration
ENABLE_RERANKING = True
ENABLE_QUERY_TRANSFORMATION = True
ENABLE_SENTENCE_WINDOW = True

# Performance Optimization Configuration
ENABLE_EMBEDDING_CACHE = True  # Cache embeddings for 30 min
ENABLE_CONNECTION_POOLING = True  # HTTP connection pooling
ENABLE_ASYNC_QUERY_TRANSFORM = True  # Parallel LLM calls

# Reranking Configuration
COHERE_API_KEY = ""
RERANK_MODEL = "rerank-english-v3.0"
RERANK_TOP_N = 3  # Reduced for faster LLM generation (less context)
INITIAL_RETRIEVAL_K = 10  # Retrieve more, rerank to top_n (reduced for performance)

# Sentence Window Configuration
SENTENCE_WINDOW_SIZE = 3  # Sentences before/after to include

# mTalkz Configuration
MTALKZ_BASE_URL = "https://rcmapi.instaalerts.zone/services/rcm/sendMessage"  # Replace with actual mTalkz API URL
MTALKZ_API_KEY = ""  # Add your mTalkz API key
MTALKZ_ACCESS_TOKEN = ""  # Add your mTalkz access token

# JWT Authentication Configuration
JWT_SECRET_KEY = "thesecrettopayismoney"  # Change this to a secure secret key
JWT_ALGORITHM = "HS256"

# PostgreSQL Configuration for Chat Memory
POSTGRES_URL = "postgresql://postgres.sacbiiolerutbchkjxrz:Sittu2711@aws-1-ap-south-1.pooler.supabase.com:5432/postgres"  # Update with your PostgreSQL connection

# Chat Memory Configuration
ENABLE_CHAT_MEMORY = True
SLIDING_WINDOW_SIZE = 20  # Keep last 20 messages in memory (10 user + 10 AI)

# =============================================================================
# Multilingual Translation Configuration
# =============================================================================
ENABLE_MULTILINGUAL = True  # Enable Hindi, English, and Hinglish support
# Translation uses the same LLM instance as the RAG pipeline (model_name from chatBotConfig)
