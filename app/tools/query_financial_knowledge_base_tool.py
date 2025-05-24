"""
Knowledge base query tool for retrieving relevant financial information.
"""
from typing import Optional
import logging

from app.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)

'''
This file contains the knowledge base tool, which is used to query the knowledge base.

It's not currently used, but could be used in the future.
'''

async def query_financial_knowledge_base_impl(
    query: str, 
    top_k: int = 5, 
    retrieval_service_instance: Optional[RetrievalService] = None
) -> str:
    """
    Query the financial knowledge base for relevant information.
    
    Args:
        query: The search query to find relevant documents
        top_k: Number of top results to return (default: 5)
        retrieval_service_instance: Optional RetrievalService instance
        
    Returns:
        Formatted string containing retrieved documents or error message
    """
    try:
        # Use provided service instance or create a new one
        if retrieval_service_instance is None:
            retrieval_service_instance = RetrievalService()
            
        # Search the knowledge base
        results = await retrieval_service_instance.search_knowledge_base(query, top_k)
        
        if not results:
            return "No relevant information found in the knowledge base for the given query."
            
        # Format the results into a coherent string
        formatted_results = ["Retrieved context from knowledge base:\n"]
        
        for i, doc in enumerate(results, 1):
            doc_id = doc.get('original_id', 'unknown')
            score = doc.get('score', 0.0)
            text_content = doc.get('text', doc.get('text_preview', 'No content available'))
            
            formatted_results.append(f"--- Document {i} (ID: {doc_id}, Score: {score:.3f}) ---")
            formatted_results.append(text_content)
            formatted_results.append("")  # Empty line for separation
            
        return "\n".join(formatted_results)
        
    except Exception as e:
        logger.error(f"Error in query_financial_knowledge_base_impl: {str(e)}")
        return f"Error retrieving information from knowledge base: {str(e)}"


# Tool schema for the financial knowledge base query function
QUERY_FINANCIAL_KB_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "query_financial_knowledge_base",
        "description": "Search the financial knowledge base for relevant information. Use this when the initially provided context is insufficient to answer the user's question. This tool will search through indexed financial documents and return the most relevant passages.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant financial information. Be specific and include key terms related to the question."
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top relevant documents to retrieve (default: 5, max: 10)",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
}