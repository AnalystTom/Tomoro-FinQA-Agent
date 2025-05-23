# app/tools/knowledge_base_tool.py
import logging
from typing import List, Dict, Any, Optional

from app.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)

QUERY_FINANCIAL_KB_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "query_financial_knowledge_base",
        "description": "Queries a financial knowledge base to find relevant information based on a natural language query. "
                       "Use this tool when the initially provided context is insufficient to answer the user's question, "
                       "and you need to search for specific financial data, definitions, or explanations.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The natural language query to search the knowledge base. "
                                   "Be specific and clear for best results. For example, 'What are the revenue recognition criteria under IFRS 15?' "
                                   "or 'Explain the concept of EBITDA'."
                },
                "top_k": {
                    "type": "integer",
                    "description": "The maximum number of relevant documents to retrieve. Defaults to 3 if not specified.",
                    "minimum": 1,
                    "maximum": 10 
                }
            },
            "required": ["query"]
        }
    }
}

async def query_financial_knowledge_base_impl(
    query: str, 
    top_k: int = 3, # Default to 3 as per schema description update
    retrieval_service_instance: Optional[RetrievalService] = None
) -> str:
    """
    Implementation of the financial knowledge base query tool.

    Args:
        query: The natural language query.
        top_k: The number of top results to retrieve.
        retrieval_service_instance: An instance of RetrievalService. If None, a new one is created.

    Returns:
        A formatted string containing the retrieved documents or a message if no results are found.
    """
    if not query:
        return "Error: Query cannot be empty."
    
    if not 1 <= top_k <= 10: # Ensure top_k is within the schema defined bounds
        logger.warning(f"top_k value {top_k} is outside the allowed range (1-10). Clamping to 3.")
        top_k = 3


    service: RetrievalService
    if retrieval_service_instance:
        service = retrieval_service_instance
    else:
        try:
            # This assumes RetrievalService can be initialized without arguments
            # and will use its default EmbeddingClient and VectorIndexer.
            service = RetrievalService()
            logger.info("Created a new RetrievalService instance for knowledge base query.")
        except Exception as e:
            logger.error(f"Failed to initialize RetrievalService: {e}", exc_info=True)
            return "Error: Could not initialize the knowledge base retrieval service."

    try:
        logger.info(f"Querying knowledge base with: '{query}', top_k={top_k}")
        documents: List[Dict[str, Any]] = await service.search_knowledge_base(query, top_k)
    except Exception as e:
        logger.error(f"Error calling search_knowledge_base: {e}", exc_info=True)
        return f"Error: An unexpected error occurred while querying the knowledge base: {str(e)}"

    if not documents:
        logger.info(f"No documents found in knowledge base for query: '{query}'")
        return "No relevant information found in the knowledge base for your query."

    # Format the results into a single string
    formatted_results = ["Retrieved context from knowledge base:\n"]
    for i, doc in enumerate(documents):
        doc_id = doc.get("id", doc.get("original_id", f"unknown_id_{i+1}"))
        score = doc.get("score", "N/A")
        text_content = doc.get("text", doc.get("text_preview", "No text content available."))
        
        # Corrected score formatting
        if isinstance(score, float):
            score_display = f"{score:.4f}"
        else:
            score_display = str(score)
        
        formatted_results.append(f"--- Document {i+1} (ID: {doc_id}, Score: {score_display}) ---")
        formatted_results.append(text_content)
        formatted_results.append("\n") # Add a newline for separation

    logger.info(f"Successfully formatted {len(documents)} documents for query: '{query}'")
    return "\n".join(formatted_results).strip()


if __name__ == '__main__':
    # Example usage for direct testing (requires mocked or real RetrievalService setup)
    async def main():
        logging.basicConfig(level=logging.INFO)
        
        # Mock RetrievalService for this example
        # Ensure this mock matches the actual RetrievalService's expected init if it changes
        class MockRetrievalService(RetrievalService):
            def __init__(self, embedding_client=None, vector_indexer=None): # Mock init
                self.embedding_client = embedding_client
                self.vector_indexer = vector_indexer
                logger.info("MockRetrievalService initialized.")
            
            async def search_knowledge_base(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
                logger.info(f"MockRetrievalService: Searching for '{query}' with top_k={top_k}")
                if "revenue recognition" in query.lower():
                    return [
                        {"id": "doc123", "score": 0.92345, "text": "IFRS 15 establishes the principles that an entity applies when reporting information about the nature, amount, timing, and uncertainty of revenue and cash flows from a contract with a customer."},
                        {"id": "doc456", "score": 0.87654, "text": "The five-step model for revenue recognition includes: 1. Identify the contract(s) with a customer. 2. Identify the performance obligations..."},
                    ]
                elif "ebitda" in query.lower():
                     return [
                        {"id": "doc789", "score": 0.95, "text": "EBITDA stands for Earnings Before Interest, Taxes, Depreciation, and Amortization. It is a measure of a company's overall financial performance."}
                     ]
                return []

        mock_service = MockRetrievalService(embedding_client=None, vector_indexer=None)

        test_queries = [
            ("What are the revenue recognition criteria under IFRS 15?", 2),
            ("Explain EBITDA", 1),
            ("Tell me about unicorns", 3) # Should find no results
        ]

        for q_text, k_val in test_queries:
            print(f"\nTesting tool with query: '{q_text}', top_k={k_val}")
            result_str = await query_financial_knowledge_base_impl(
                query=q_text, 
                top_k=k_val,
                retrieval_service_instance=mock_service
            )
            print("Tool Output:")
            print(result_str)
            print("-" * 30)
        
        print("\nTesting schema:")
        import json
        print(json.dumps(QUERY_FINANCIAL_KB_TOOL_SCHEMA, indent=2))

    import asyncio
    # This requires app.core.config.settings to be available for the real RetrievalService initialization
    # if query_financial_knowledge_base_impl is called without a retrieval_service_instance.
    # For the __main__ block, we are providing a mock instance, so it should be fine.
    # However, ensure your environment is set up for 'from app.core.config import settings' to work
    # when RetrievalService() is instantiated directly.
    
    # A minimal mock for settings if needed for direct execution and RetrievalService() is called
    class MockSettingsGlobal:
        EMBEDDING_MODEL_NAME = "mock-model-global"
        VECTOR_INDEX_PATH = "mock_index_global.faiss"
        EMBEDDING_DIMENSION = 10 

    # To make RetrievalService() work if called without instance in a standalone script execution:
    # You would typically have your app.core.config.py define and instantiate settings.
    # For example, app.core.config.py might have:
    # class Settings(BaseSettings): ...
    # settings = Settings()
    # If running this script directly and `app.core.config.settings` isn't resolvable,
    # you might need to mock it globally for the test, e.g., by adding a mock config.py to sys.modules
    # or ensuring your PYTHONPATH is correct.
    # The current __main__ uses a MockRetrievalService, so it bypasses direct RetrievalService() instantiation.

    asyncio.run(main())
