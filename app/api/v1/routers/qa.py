# app/api/v1/routers/qa.py
import logging
from fastapi import APIRouter, HTTPException, Depends

# Import the updated schemas from the Canvas
from app.api.v1.schemas.qa_schemas import QAInput, QAResponse 

# Import the actual QAService (which now has process_conversation_turn)
# The QAService itself is from the Canvas (ID: qa_service_py_multiturn)
from app.services.qa_service import QAService

logger = logging.getLogger(__name__)

router = APIRouter()

# Dependency getter for the QAService
def get_qa_service() -> QAService:
    """
    Dependency injector for QAService.
    Initializes and returns an instance of QAService.
    """
    try:
        # QAService might raise RuntimeError if its own dependencies fail to init
        return QAService()
    except RuntimeError as e:
        logger.error(f"Fatal error initializing QAService for dependency: {e}", exc_info=True)
        # This error will be caught by FastAPI's default exception handling or a custom one
        # if this function is called during app startup for dependency checks.
        # If called per-request, the endpoint's try-except might handle it too.
        # Raising HTTPException here makes it explicit for this dependency.
        raise HTTPException(status_code=503, detail=f"Q&A Service is unavailable due to initialization error: {e}")
    except Exception as e: 
        logger.error(f"Unexpected fatal error initializing QAService for dependency: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Q&A Service is critically unavailable: {e}")


@router.post(
    "/process-query", # Endpoint path remains the same for now
    response_model=QAResponse, # Uses the updated QAResponse with updated_messages_history
    summary="Process a Question or Conversation Turn with Context",
    description=(
        "Accepts a question (current user utterance) and optionally an existing conversation history (`messages_history`). "
        "For the first turn of a conversation, `pre_text`, `post_text`, and `table_ori` can be provided to establish initial context. "
        "For subsequent turns, the `messages_history` should be passed back from the previous response, and the `question` field "
        "should contain the new user input."
    )
)
async def process_conversation_turn_endpoint( # Renamed endpoint function for clarity
    qa_input: QAInput, # Uses the updated QAInput with messages_history
    qa_service: QAService = Depends(get_qa_service) 
):
    """
    Processes a single turn in a financial Q&A conversation.

    - **qa_input**: Contains the current question/utterance.
        - For the *first turn*, it can also include `pre_text`, `post_text`, `table_ori` to set initial context. `messages_history` would be null or empty.
        - For *subsequent turns*, it should include `messages_history` from the previous turn's response.
    """
    logger.info(
        f"Received request for /process-query. Question: '{qa_input.question[:50]}...'. "
        f"History provided: {'Yes' if qa_input.messages_history else 'No'} "
        f"(Item ID: {qa_input.item_id}, Request ID: {qa_input.request_id})"
    )
    try:
        # Call the updated service method
        response = await qa_service.process_conversation_turn(qa_input)
        
        # Set item_id and request_id in the response if they were in the input, for client tracking
        # QAResponse model already has these fields as optional.
        response.item_id = qa_input.item_id
        response.request_id = qa_input.request_id
        
        return response
    except HTTPException as http_exc:
        # Re-raise HTTPExceptions that might have been raised by get_qa_service or elsewhere
        logger.error(f"HTTPException in /process-query: {http_exc.status_code} - {http_exc.detail}", exc_info=False) # No need for full exc_info for HTTPExceptions
        raise http_exc
    except Exception as e:
        # This catches unexpected errors from qa_service.process_conversation_turn
        # or other unforeseen issues within this endpoint.
        logger.error(
            f"Unexpected error in /process-query for question '{qa_input.question[:50]}...' "
            f"(Item ID: {qa_input.item_id}): {str(e)}", exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred while processing your query: {str(e)}")

