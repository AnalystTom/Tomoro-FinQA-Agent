# app/api/v1/routers/qa.py
import logging
from fastapi import APIRouter, HTTPException, Depends
from app.api.v1.schemas.qa_schemas import QAInput, QAResponse

# Import the actual QAService
from app.services.qa_service import QAService

logger = logging.getLogger(__name__)

router = APIRouter()

# Dependency getter for the actual QAService
def get_qa_service() -> QAService:
    """
    Dependency injector for QAService.
    Initializes and returns an instance of QAService.
    In a more complex application, this could handle singleton instances,
    or retrieve a service configured with more specific dependencies.
    """
    try:
        return QAService()
    except RuntimeError as e:
        # If QAService initialization fails (e.g., critical component missing),
        # this will prevent the app from starting or requests from being processed
        # if the dependency can't be fulfilled.
        logger.error(f"Fatal error initializing QAService: {e}", exc_info=True)
        # Depending on FastAPI version and setup, raising an error here might
        # prevent app startup or cause 500 errors on requests needing this dependency.
        # For robustness in request handling, one might catch this in the endpoint
        # or ensure QAService handles its init failures more gracefully if possible.
        # However, a failing service init is usually a critical issue.
        raise HTTPException(status_code=503, detail=f"Q&A Service is unavailable due to initialization error: {e}")
    except Exception as e: # Catch any other unexpected init errors
        logger.error(f"Unexpected fatal error initializing QAService: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Q&A Service is critically unavailable: {e}")


@router.post(
    "/process-query",
    response_model=QAResponse,
    summary="Process a Question with Provided Context",
    description="Accepts a question along with its surrounding text (pre_text, post_text) and a table (table_ori). "
                "This endpoint is primarily designed for scenarios like ConvFinQA where the immediate context for "
                "the question is already identified and provided."
)
async def process_query_endpoint(
    qa_input: QAInput,
    qa_service: QAService = Depends(get_qa_service) # Use the real QAService
):
    """
    Processes a single financial question based on the provided context (pre-text, post-text, and table_ori).

    - **qa_input**: Contains the question and its associated contextual data.
    """
    logger.info(f"Received request for /process-query. Question snippet: {qa_input.question[:50]}...")
    try:
        response = await qa_service.process_single_entry_query(qa_input)
        # The QAResponse model itself will be validated by FastAPI.
        # The real QAService incorporates error details into the answer/explanation
        # or would have raised an exception leading to an HTTP error response.
        return response
    except HTTPException as http_exc:
        # Re-raise HTTPExceptions that might have been raised by get_qa_service or elsewhere
        logger.error(f"HTTPException in /process-query: {http_exc.status_code} - {http_exc.detail}", exc_info=True)
        raise http_exc
    except Exception as e:
        # This catches unexpected errors from qa_service.process_single_entry_query
        # or other unforeseen issues within this endpoint.
        logger.error(f"Unexpected error in /process-query for question '{qa_input.question[:50]}...': {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred while processing your query: {str(e)}")

# Note: You might add another endpoint later, e.g., /general-query,
# which would take only a question and use the full RAG pipeline (including Stage 1's vector store).
