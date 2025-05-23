# app/core/config.py
import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env file if it exists
# This is useful for local development.
# In production, environment variables should be set directly.
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env') # Points to project root .env
load_dotenv(dotenv_path=dotenv_path)

class Settings(BaseSettings):
    """
    Application settings are loaded from environment variables.
    For local development, you can use a .env file in the project root.
    """
    # --- Embedding Client Configuration ---
    # Type of embedding client to use: "openai" or "local"
    EMBEDDING_CLIENT_TYPE: str = "local"

    # --- OpenAI API Configuration (Used if EMBEDDING_CLIENT_TYPE is "openai", or for LLM) ---
    OPENAI_API_KEY: Optional[str] = "YOUR_OPENAI_API_KEY_HERE"  # Default if not set in env

    # --- Model Names ---
    # Embedding model used for generating vector embeddings
    # If EMBEDDING_CLIENT_TYPE is "openai", this is an OpenAI model name (e.g., "text-embedding-3-small")
    # If EMBEDDING_CLIENT_TYPE is "local", this is a SentenceTransformer model name from Hugging Face
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

    # Large Language Model used for Q&A and reasoning (currently assumes OpenAI)
    LLM_MODEL_NAME: str = "gpt-4o" # Or "gpt-3.5-turbo", "gpt-4-turbo-preview" etc.

    # --- Local Embedding Model Configuration (Used if EMBEDDING_CLIENT_TYPE is "local") ---
    # Device for local SentenceTransformer model: "cpu", "cuda", "mps" etc.
    LOCAL_EMBEDDING_DEVICE: str = "cpu"
    # Path to cache downloaded SentenceTransformer models (optional)
    SENTENCE_TRANSFORMERS_HOME: Optional[str] = None


    # --- Data Paths ---
    # Path to the training JSON file (e.g., train.json from ConvFinQA)
    TRAIN_JSON_PATH: str = "data/raw_documents/train.json"
    # Path where the vector store (e.g., FAISS index, ChromaDB data) is/will be saved
    VECTOR_STORE_PATH: str = "data/vector_store"
    VECTOR_INDEX_PATH: str = "data/vector_store/index.faiss"
    # Path for application logs
    LOG_FILE_PATH: str = "data/logs/app.log"
    MAX_TOOL_ITERATIONS: int = 5
    # --- Vector Store Configuration ---
    # Type of vector database to use (e.g., "faiss", "chromadb")
    VECTOR_DB_TYPE: str = "faiss"
    # Dimension of the embeddings. This MUST match the output dimension of your chosen EMBEDDING_MODEL_NAME.
    # OpenAI "text-embedding-3-small": 1536
    # OpenAI "text-embedding-ada-002": 1536
    # SentenceTransformer "all-MiniLM-L6-v2": 384
    # SentenceTransformer "all-mpnet-base-v2": 768
    EMBEDDING_DIMENSION: int = 384 # Default for "all-MiniLM-L6-v2"

    # --- Application Behavior ---
    # Logging level for the application (e.g., INFO, DEBUG, WARNING, ERROR)
    LOG_LEVEL: str = "INFO"
    # Default number of top-k documents to retrieve in RAG
    DEFAULT_RETRIEVAL_TOP_K: int = 5

    # --- API Settings (If you're building an API) ---
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Financial AI Agent"
    PROJECT_VERSION: str = "0.1.0"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = 'ignore'

# Instantiate settings
settings = Settings()

# --- Example .env file content (create a .env file in your project root) ---
"""
# .env example (place this in your project root directory, NOT in app/core/)

# --- Embedding Client ---
EMBEDDING_CLIENT_TYPE="local"

# --- OpenAI Configuration (Still needed for LLM_MODEL_NAME if using OpenAI for that) ---
OPENAI_API_KEY="sk-yourActualOpenAIapiKeyGoesHere" # Keep for LLM, or set to dummy if only using local embeddings and no OpenAI LLM

# --- Model Names ---
EMBEDDING_MODEL_NAME="all-MiniLM-L6-v2" # Example SentenceTransformer model
# LLM_MODEL_NAME="gpt-4o" # Stays as OpenAI model for now

# --- Local Embedding Model Configuration ---
LOCAL_EMBEDDING_DEVICE="cpu" # Or "cuda" if you have a GPU and PyTorch with CUDA
# SENTENCE_TRANSFORMERS_HOME="/path/to/your/cache/dir" # Optional: if you want to control model download location

# --- Vector Store Configuration ---
EMBEDDING_DIMENSION=384 # IMPORTANT: This matches "all-MiniLM-L6-v2" (which is 384)

# --- Other settings from before ---
# TRAIN_JSON_PATH="data/raw_documents/train.json"
# VECTOR_STORE_PATH="data/vector_store"
# LOG_FILE_PATH="data/logs/app.log"
# VECTOR_DB_TYPE="faiss"
# LOG_LEVEL="DEBUG"
# DEFAULT_RETRIEVAL_TOP_K=5
"""
