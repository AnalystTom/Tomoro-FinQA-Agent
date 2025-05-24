# app/api/v1/schemas/qa_schemas.py
from pydantic import BaseModel, Field
from typing import List, Any, Optional, Dict, Union

class ToolCallLog(BaseModel):
    tool_name: str
    tool_args: Dict[str, Any] # Expecting parsed arguments as a dict
    tool_result: str          # Result of the tool call, or error message
    error: bool = False       # Flag to indicate if the tool call resulted in an error

class QAInput(BaseModel):
    question: str  # For the first turn, this is the main question. For subsequent turns, the current user utterance.
    
    # Context fields, primarily for the first turn of a conversation
    pre_text: Optional[List[str]] = None
    post_text: Optional[List[str]] = None
    table_ori: Optional[List[List[Any]]] = None # List of lists, or other raw table format

    # Optional field for existing conversation history
    # Each dict in the list should conform to OpenAI's message format
    # e.g., {"role": "user", "content": "..."} or {"role": "assistant", "content": "...", "tool_calls": [...]}
    messages_history: Optional[List[Dict[str, Any]]] = Field(default=None, description="Existing conversation history.")

    # Optional identifiers
    uid: Optional[str] = None
    request_id: Optional[str] = None
    item_id: Optional[str] = None # Useful for evaluation linking

class QAResponse(BaseModel):
    answer: str  # The final textual answer from the assistant for the current turn
    explanation: Optional[str] = None
    tool_calls_log: List[ToolCallLog] = Field(default_factory=list) # Log of tools used in the *current* processing step
    
    # The complete, updated conversation history including the latest turn
    updated_messages_history: List[Dict[str, Any]] = Field(description="The full conversation history after this turn.")
    
    # Optional: You might want to include the original item_id or request_id back for client tracking
    item_id: Optional[str] = None
    request_id: Optional[str] = None

