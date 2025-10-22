from pymongo import MongoClient
from config.settings import MONGODB_URL, DATABASE_NAME

# MongoDB client - READ ONLY DATABASE
# WARNING: This database is for READ operations only. Do NOT write, update, or delete any records.
mongo_client = MongoClient(MONGODB_URL)
db = mongo_client[DATABASE_NAME]

# Collections for READ ONLY access
students_collection = db["students"]  # READ ONLY
users_collection = db["users"]  # READ ONLY
communication_configs_collection = db["communicationconfigs"]  # READ ONLY
entities_collection = db["entities"]  # READ ONLY
