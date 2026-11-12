import os
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

# API Keys and database URI
JINA_API_KEY = os.getenv("JINA_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")

# MongoDB configuration
DATABASE_NAME = "rag_database"
COLLECTION_NAME = "documents_jina"

# Cloud AI Models
EMBEDDING_MODEL = "jina-embeddings-v3"
LLM_MODEL = "qwen/qwen3.8-27b"

# Embedding and vector search settings
EMBEDDING_DIMENSIONS = 1024
VECTOR_INDEX_NAME = "vector_index_jina"
SIMILARITY_THRESHOLD = 0.59

# Legacy settings preserved for rollback
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OLD_COLLECTION_NAME = "documents"
OLD_VECTOR_INDEX_NAME = "vector_index"
OLD_EMBEDDING_DIMENSIONS = 3072
