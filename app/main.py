# app/main.py
import logging
from fastapi import FastAPI
from app.api.v1.routers import qa as qa_router_v1
from app.config.settings import settings # Your application settings

# Configure logging
# You can make this more sophisticated (e.g., based on settings.LOG_LEVEL)
logging.basicConfig(
    level=settings.LOG_LEVEL.upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler() # Log to console
        # Optionally add FileHandler if LOG_FILE_PATH is set
        # logging.FileHandler(settings.LOG_FILE_PATH)
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    description="A Financial AI Agent for Q&A on financial documents using RAG and Agentic capabilities.",
    # You can add more metadata like terms_of_service, contact, license_info
)

# Include the v1 Q&A router
# The prefix here means all routes in qa_router_v1 will start with /api/v1/qa
app.include_router(qa_router_v1.router, prefix=f"{settings.API_V1_STR}/qa", tags=["Q&A Endpoints"])

@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting up {settings.PROJECT_NAME} v{settings.PROJECT_VERSION}...")
    logger.info(f"Log level set to: {settings.LOG_LEVEL}")
    # You can add other startup logic here, e.g., initializing ML models, DB connections if not done lazily.

@app.on_event("shutdown")
async def shutdown_event():
    logger.info(f"Shutting down {settings.PROJECT_NAME}...")
    # Add cleanup logic here if needed

@app.get("/", tags=["Root"])
async def read_root():
    """
    Root endpoint providing a welcome message.
    """
    return {"message": f"Welcome to the {settings.PROJECT_NAME}! Visit /docs for API documentation."}

# To run this application (from the project root directory):
# Ensure your .env file is set up, especially OPENAI_API_KEY if QAService eventually uses it.
# Command: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# (Adjust host and port as needed, e.g., from settings.API_HOST, settings.API_PORT)
