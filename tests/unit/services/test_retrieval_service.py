# tests/unit/services/test_retrieval_service.py
import unittest
from unittest.mock import MagicMock, AsyncMock, patch
import numpy as np
from typing import List, Dict, Any

# Ensure app path is available or adjust imports
from app.services.retrieval_service import RetrievalService
from app.clients.embedding_client import EmbeddingClient
from app.rag_pipeline.indexer import VectorIndexer
from app.config.settings import settings # Required for default initialization

class TestRetrievalService(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        # Mock the clients that RetrievalService depends on
        self.mock_embedding_client = AsyncMock(spec=EmbeddingClient)
        self.mock_vector_indexer = MagicMock(spec=VectorIndexer)

        # Configure the vector indexer mock
        self.mock_vector_indexer.is_index_available.return_value = True # Assume index is available by default
        self.mock_vector_indexer.load_index = MagicMock() # Mock load_index method
        
        # Patch settings to avoid issues with real paths/configs during tests
        # We are providing mocks directly, but default init might still try to use settings
        self.settings_patcher = patch('app.services.retrieval_service.settings', autospec=True)
        self.mock_settings = self.settings_patcher.start()
        
        # Provide some default mock values for settings if RetrievalService tries to use them
        self.mock_settings.EMBEDDING_MODEL_NAME = "test-embed-model"
        self.mock_settings.VECTOR_INDEX_PATH = "/fake/path/index.faiss"
        self.mock_settings.EMBEDDING_DIMENSION = 768 # Example dimension

        # Instantiate RetrievalService with mocked dependencies
        self.retrieval_service = RetrievalService(
            embedding_client=self.mock_embedding_client,
            vector_indexer=self.mock_vector_indexer
        )

    def tearDown(self):
        self.settings_patcher.stop()

    async def test_search_knowledge_base_success(self):
        """Test successful search_knowledge_base operation."""
        query = "What is revenue?"
        top_k = 3
        mock_embedding = np.array([0.1, 0.2, 0.3] * (settings.EMBEDDING_DIMENSION // 3) ).astype('float32') # Use actual dimension
        
        # Mock EmbeddingClient's response
        self.mock_embedding_client.generate_embeddings.return_value = [mock_embedding.tolist()]

        # Mock VectorIndexer's response
        # This should be a list of dictionaries as per the plan
        mock_search_results = [
            {"id": "doc1", "text_preview": "Revenue is income...", "score": 0.9, "metadata": {"source": "doc_source1"}},
            {"id": "doc2", "text_preview": "Understanding revenue streams...", "score": 0.8, "metadata": {"source": "doc_source2"}},
        ]
        self.mock_vector_indexer.search.return_value = mock_search_results

        results = await self.retrieval_service.search_knowledge_base(query, top_k)

        self.mock_embedding_client.generate_embeddings.assert_called_once_with([query])
        
        # Check that the query embedding passed to search is a 2D numpy array
        self.mock_vector_indexer.search.assert_called_once()
        call_args = self.mock_vector_indexer.search.call_args
        passed_embedding = call_args[0][0]
        self.assertIsInstance(passed_embedding, np.ndarray)
        self.assertEqual(passed_embedding.ndim, 2) # Should be (1, dim)
        self.assertEqual(passed_embedding.shape[1], settings.EMBEDDING_DIMENSION)
        np.testing.assert_array_almost_equal(passed_embedding[0], mock_embedding, decimal=5)
        self.assertEqual(call_args[1]['top_k'], top_k) # Check top_k argument

        self.assertEqual(len(results), 2)
        self.assertEqual(results, mock_search_results)
        self.mock_vector_indexer.load_index.assert_not_called() # Should not be called if index is available

    async def test_search_knowledge_base_empty_query(self):
        results = await self.retrieval_service.search_knowledge_base("", top_k=3)
        self.assertEqual(results, [])
        self.mock_embedding_client.generate_embeddings.assert_not_called()
        self.mock_vector_indexer.search.assert_not_called()

    async def test_search_knowledge_base_top_k_invalid(self):
        # Test with top_k = 0, should default to 5 (or a predefined default)
        await self.retrieval_service.search_knowledge_base("query", top_k=0)
        # The service logs a warning and defaults top_k. Check if search was called with default.
        # Assuming default is 5 as per current implementation detail.
        self.mock_vector_indexer.search.assert_called_once()
        self.assertEqual(self.mock_vector_indexer.search.call_args[1]['top_k'], 5)

        self.mock_vector_indexer.search.reset_mock()
        await self.retrieval_service.search_knowledge_base("query", top_k=-1)
        self.mock_vector_indexer.search.assert_called_once()
        self.assertEqual(self.mock_vector_indexer.search.call_args[1]['top_k'], 5)


    async def test_search_knowledge_base_embedding_failure(self):
        self.mock_embedding_client.generate_embeddings.return_value = [] # Simulate embedding failure
        results = await self.retrieval_service.search_knowledge_base("query", top_k=3)
        self.assertEqual(results, [])
        self.mock_vector_indexer.search.assert_not_called()

    async def test_search_knowledge_base_search_failure(self):
        mock_embedding = np.array([0.1, 0.2, 0.3] * (settings.EMBEDDING_DIMENSION // 3)).astype('float32')
        self.mock_embedding_client.generate_embeddings.return_value = [mock_embedding.tolist()]
        self.mock_vector_indexer.search.side_effect = Exception("Vector DB search failed")
        
        results = await self.retrieval_service.search_knowledge_base("query", top_k=3)
        self.assertEqual(results, [])

    async def test_search_knowledge_base_index_not_available_initially_then_loads(self):
        query = "test query"
        top_k = 2
        mock_embedding = np.array([0.1] * settings.EMBEDDING_DIMENSION).astype('float32')
        self.mock_embedding_client.generate_embeddings.return_value = [mock_embedding.tolist()]
        
        mock_search_results = [{"id": "doc1", "score": 0.9, "text_preview": "content"}]
        
        # Simulate index not available first, then available after load_index
        self.mock_vector_indexer.is_index_available.side_effect = [False, True] 
        self.mock_vector_indexer.search.return_value = mock_search_results

        results = await self.retrieval_service.search_knowledge_base(query, top_k)

        self.assertEqual(self.mock_vector_indexer.is_index_available.call_count, 2)
        self.mock_vector_indexer.load_index.assert_called_once()
        self.mock_vector_indexer.search.assert_called_once()
        self.assertEqual(results, mock_search_results)

    async def test_search_knowledge_base_index_not_available_and_load_fails(self):
        query = "test query"
        mock_embedding = np.array([0.1] * settings.EMBEDDING_DIMENSION).astype('float32')
        self.mock_embedding_client.generate_embeddings.return_value = [mock_embedding.tolist()]
        
        # Simulate index not available, and load_index fails to make it available
        self.mock_vector_indexer.is_index_available.side_effect = [False, False] 
        self.mock_vector_indexer.load_index.side_effect = Exception("Failed to load index file")

        results = await self.retrieval_service.search_knowledge_base(query, top_k=2)

        self.assertEqual(self.mock_vector_indexer.is_index_available.call_count, 2) # Initial check, then check after load attempt
        self.mock_vector_indexer.load_index.assert_called_once()
        self.mock_vector_indexer.search.assert_not_called()
        self.assertEqual(results, [])
    
    # Test default initialization if mocks are not passed (more complex to set up fully)
    @patch('app.services.retrieval_service.EmbeddingClient')
    @patch('app.services.retrieval_service.VectorIndexer')
    async def test_default_initialization(self, MockVI, MockEC):
        # Mock the constructors of the clients
        mock_ec_instance = AsyncMock(spec=EmbeddingClient)
        mock_vi_instance = MagicMock(spec=VectorIndexer)
        mock_vi_instance.is_index_available.return_value = False # Say index not initially loaded
        mock_vi_instance.load_index = MagicMock()


        MockEC.return_value = mock_ec_instance
        MockVI.return_value = mock_vi_instance
        
        service = RetrievalService() # Initialize without passing clients

        self.assertIsNotNone(service.embedding_client)
        self.assertIsNotNone(service.vector_indexer)
        MockEC.assert_called_once_with(model_name=self.mock_settings.EMBEDDING_MODEL_NAME)
        MockVI.assert_called_once_with(index_path=self.mock_settings.VECTOR_INDEX_PATH, embedding_dimension=self.mock_settings.EMBEDDING_DIMENSION)
        mock_vi_instance.load_index.assert_called_once() # Default init tries to load index

if __name__ == '__main__':
    unittest.main()
