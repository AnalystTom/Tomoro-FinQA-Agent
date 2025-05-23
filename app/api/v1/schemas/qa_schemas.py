# app/api/v1/schemas/qa_schemas.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class QAInput(BaseModel):
    question: str
    pre_text: Optional[List[str]] = None       # Allows None, defaults to None if not provided
    post_text: Optional[List[str]] = None      # Allows None, defaults to None if not provided

    table_ori: Optional[List[List[Any]]] = None # Allows None, defaults to None


class ToolCallLog(BaseModel):
    tool_name: str = Field(..., description="Name of the tool called.", examples=["calculator"])
    tool_args: Dict[str, Any] = Field(..., description="Arguments passed to the tool.", examples=[{"math_expression": "100 + 50"}])
    tool_result: Any = Field(..., description="Result returned by the tool.", examples=["150"]) # Can be string, number, dict, list

class QAResponse(BaseModel):
    answer: str = Field(..., description="The final answer to the question.", examples=["The revenue in 2023 was 100M."])
    explanation: Optional[str] = Field(None, description="Explanation of how the answer was derived.", examples=["The value was found in the provided table."])
    tool_calls_log: Optional[List[ToolCallLog]] = Field(None, description="Log of tool calls made by the agent.")
    retrieved_context_summary: Optional[str] = Field(None, description="Summary of context retrieved by the 'query_financial_knowledge_base' tool, if used.")
    error_message: Optional[str] = Field(None, description="Error message if processing failed at some point.") # Ensure this line is present

