# tests/unit/tools/test_knowledge_base_tool.py
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from app.tools.knowledge_base_tool import query_financial_knowledge_base_impl, QUERY_FINANCIAL_KB_TOOL_SCHEMA
from app.services.retrieval_service import RetrievalService # For type hinting and mocking

class TestQueryFinancialKnowledgeBaseTool(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.mock_retrieval_service = AsyncMock(spec=RetrievalService)

    async def test_query_kb_success_with_results(self):
        query = "What is IFRS 15?"
        top_k = 2
        mock_docs = [
            {"id": "doc1", "score": 0.95, "text": "IFRS 15 details revenue recognition."},
            {"id": "doc2", "score": 0.88, "text": "It involves a five-step model."}
        ]
        self.mock_retrieval_service.search_knowledge_base.return_value = mock_docs

        result_str = await query_financial_knowledge_base_impl(
            query, top_k, retrieval_service_instance=self.mock_retrieval_service
        )

        self.mock_retrieval_service.search_knowledge_base.assert_called_once_with(query, top_k)
        self.assertIn("Retrieved context from knowledge base:", result_str)
        self.assertIn("--- Document 1 (ID: doc1, Score: 0.9500) ---", result_str)
        self.assertIn("IFRS 15 details revenue recognition.", result_str)
        self.assertIn("--- Document 2 (ID: doc2, Score: 0.8800) ---", result_str)
        self.assertIn("It involves a five-step model.", result_str)

    async def test_query_kb_no_results_found(self):
        query = "Tell me about obscure financial term X."
        top_k = 3
        self.mock_retrieval_service.search_knowledge_base.return_value = [] # No documents

        result_str = await query_financial_knowledge_base_impl(
            query, top_k, retrieval_service_instance=self.mock_retrieval_service
        )

        self.mock_retrieval_service.search_knowledge_base.assert_called_once_with(query, top_k)
        self.assertEqual(result_str, "No relevant information found in the knowledge base for your query.")

    async def test_query_kb_empty_query_string(self):
        result_str = await query_financial_knowledge_base_impl(
            "", 2, retrieval_service_instance=self.mock_retrieval_service
        )
        self.assertEqual(result_str, "Error: Query cannot be empty.")
        self.mock_retrieval_service.search_knowledge_base.assert_not_called()

    async def test_query_kb_top_k_out_of_bounds(self):
        query = "Valid query"
        # Test top_k too low
        await query_financial_knowledge_base_impl(query, 0, self.mock_retrieval_service)
        self.mock_retrieval_service.search_knowledge_base.assert_called_once_with(query, 3) # Should clamp to default 3
        self.mock_retrieval_service.search_knowledge_base.reset_mock()

        # Test top_k too high
        await query_financial_knowledge_base_impl(query, 15, self.mock_retrieval_service)
        self.mock_retrieval_service.search_knowledge_base.assert_called_once_with(query, 3) # Should clamp to default 3
    
    async def test_query_kb_retrieval_service_search_raises_exception(self):
        query = "A query that causes trouble."
        top_k = 1
        self.mock_retrieval_service.search_knowledge_base.side_effect = Exception("DB connection lost")

        result_str = await query_financial_knowledge_base_impl(
            query, top_k, retrieval_service_instance=self.mock_retrieval_service
        )
        
        self.assertIn("Error: An unexpected error occurred while querying the knowledge base: DB connection lost", result_str)

    @patch('app.tools.knowledge_base_tool.RetrievalService')
    async def test_query_kb_no_retrieval_service_instance_provided(self, MockRetrievalServiceClass):
        # Mock the RetrievalService class constructor and its instance's methods
        mock_service_instance = AsyncMock(spec=RetrievalService)
        mock_service_instance.search_knowledge_base.return_value = [
            {"id": "docA", "score": 0.7, "text": "Content from default service."}
        ]
        MockRetrievalServiceClass.return_value = mock_service_instance
        
        query = "Test with default service"
        top_k = 1
        result_str = await query_financial_knowledge_base_impl(query, top_k, retrieval_service_instance=None)

        MockRetrievalServiceClass.assert_called_once() # Check if RetrievalService() was called
        mock_service_instance.search_knowledge_base.assert_called_once_with(query, top_k)
        self.assertIn("Content from default service.", result_str)

    @patch('app.tools.knowledge_base_tool.RetrievalService')
    async def test_query_kb_retrieval_service_init_fails(self, MockRetrievalServiceClass):
        MockRetrievalServiceClass.side_effect = Exception("Failed to init service")

        query = "Test service init failure"
        top_k = 1
        result_str = await query_financial_knowledge_base_impl(query, top_k, retrieval_service_instance=None)
        
        self.assertEqual(result_str, "Error: Could not initialize the knowledge base retrieval service.")


    def test_query_financial_kb_tool_schema_structure(self):
        self.assertIsInstance(QUERY_FINANCIAL_KB_TOOL_SCHEMA, dict)
        self.assertEqual(QUERY_FINANCIAL_KB_TOOL_SCHEMA["type"], "function")
        
        function_def = QUERY_FINANCIAL_KB_TOOL_SCHEMA["function"]
        self.assertEqual(function_def["name"], "query_financial_knowledge_base")
        self.assertIn("description", function_def)
        
        parameters = function_def["parameters"]
        self.assertEqual(parameters["type"], "object")
        self.assertIn("properties", parameters)
        
        # Check 'query' parameter
        self.assertIn("query", parameters["properties"])
        self.assertEqual(parameters["properties"]["query"]["type"], "string")
        self.assertIn("description", parameters["properties"]["query"])
        
        # Check 'top_k' parameter
        self.assertIn("top_k", parameters["properties"])
        self.assertEqual(parameters["properties"]["top_k"]["type"], "integer")
        self.assertIn("description", parameters["properties"]["top_k"])
        self.assertEqual(parameters["properties"]["top_k"]["minimum"], 1)
        self.assertEqual(parameters["properties"]["top_k"]["maximum"], 10)
        
        self.assertEqual(parameters["required"], ["query"])

if __name__ == '__main__':
    unittest.main()
