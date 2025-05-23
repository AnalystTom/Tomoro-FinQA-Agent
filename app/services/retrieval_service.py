# app/services/retrieval_service.py
import logging
from typing import List, Dict, Any, Optional
import numpy as np

from app.clients.embedding_client import EmbeddingClient # Assuming this structure
from app.rag_pipeline.indexer import VectorIndexer # Assuming this structure
from app.config.settings import settings # Assuming settings are available

logger = logging.getLogger(__name__)

class RetrievalService:
    """
    Service for retrieving documents from a vector store.
    """
    def __init__(self, 
                 embedding_client: Optional[EmbeddingClient] = None, 
                 vector_indexer: Optional[VectorIndexer] = None):
        """
        Initializes the RetrievalService.

        Args:
            embedding_client: An instance of EmbeddingClient. If None, a default is created.
            vector_indexer: An instance of VectorIndexer. If None, a default is created.
        """
        if embedding_client is None:
            # This assumes EmbeddingClient can be initialized without arguments
            # or uses global settings. Adjust if specific config is needed.
            self.embedding_client = EmbeddingClient(
                model_name=settings.EMBEDDING_MODEL_NAME,
                # Add other necessary parameters from settings if any
            ) 
            logger.info(f"Initialized EmbeddingClient with model: {settings.EMBEDDING_MODEL_NAME}")
        else:
            self.embedding_client = embedding_client

        if vector_indexer is None:
            # This assumes VectorIndexer can be initialized.
            # It might need parameters like index_path, dimension from settings.
            self.vector_indexer = VectorIndexer(
                index_path=settings.VECTOR_INDEX_PATH, 
                embedding_dimension=settings.EMBEDDING_DIMENSION
            )
            # Attempt to load the index if it exists
            try:
                self.vector_indexer.load_index()
                logger.info(f"VectorIndexer loaded index from: {settings.VECTOR_INDEX_PATH}")
            except FileNotFoundError:
                logger.warning(f"No pre-existing index found at {settings.VECTOR_INDEX_PATH}. VectorIndexer initialized empty.")
            except Exception as e:
                logger.error(f"Error loading index for VectorIndexer from {settings.VECTOR_INDEX_PATH}: {e}", exc_info=True)
                # Depending on desired behavior, could raise e or continue with an empty indexer
        else:
            self.vector_indexer = vector_indexer
        
        logger.info("RetrievalService initialized.")

    async def search_knowledge_base(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches the knowledge base for documents relevant to the query.

        Args:
            query: The user query string.
            top_k: The number of top results to retrieve.

        Returns:
            A list of dictionaries, each representing a retrieved document
            with its metadata and score.
        """
        if not query:
            logger.warning("Search query is empty.")
            return []
        if top_k <= 0:
            logger.warning(f"top_k must be positive, got {top_k}. Defaulting to 5.")
            top_k = 5

        try:
            logger.debug(f"Generating embedding for query: '{query}'")
            # EmbeddingClient.generate_embeddings expects a list of texts and returns a list of embeddings
            query_embedding_list = await self.embedding_client.generate_embeddings([query])
            if not query_embedding_list or query_embedding_list[0] is None:
                logger.error("Failed to generate embedding for the query.")
                return []
            
            query_embedding_np = np.array(query_embedding_list[0]).astype('float32')
            if query_embedding_np.ndim == 1:
                 query_embedding_np = np.expand_dims(query_embedding_np, axis=0)


            logger.debug(f"Searching vector index with top_k={top_k}")
            # VectorIndexer.search should return distances and indices/IDs
            # And then VectorIndexer.get_documents_by_ids should fetch the actual content
            # For simplicity in this stage, let's assume search directly returns document-like dicts
            # or that VectorIndexer has a method that combines search and metadata retrieval.
            # The plan mentions: "search method ... should return a list of dictionaries, 
            # each containing document metadata ... and a score."
            
            # Ensure the index is loaded/available
            if not self.vector_indexer.is_index_available():
                logger.error("Vector index is not available for searching.")
                # Attempt to load it again, or handle as a persistent error
                try:
                    self.vector_indexer.load_index()
                    if not self.vector_indexer.is_index_available():
                         logger.error("Failed to load vector index on demand.")
                         return []
                except Exception as e:
                    logger.error(f"Failed to load vector index on demand: {e}")
                    return []

            results = self.vector_indexer.search(query_embedding_np, top_k=top_k)
            
            # `results` is expected to be List[Tuple[float, Dict[str, Any]]] (score, metadata_dict)
            # or List[Dict[str, Any]] where dict includes 'score' and 'text'/'original_id' etc.
            # Let's assume the latter based on the plan's description for `search` method.
            
            # Example processing if search returns (distances, ids) and we need to fetch docs:
            # distances, doc_ids = self.vector_indexer.search_raw(query_embedding_np, top_k=top_k)
            # documents = self.vector_indexer.get_documents_by_ids(doc_ids)
            # formatted_results = []
            # for i, doc in enumerate(documents):
            #     formatted_results.append({
            #         "id": doc_ids[i], # or doc.get('original_id')
            #         "score": 1 - distances[i] if distances are cosine distances, else distances[i], # Adjust score interpretation
            #         "text": doc.get('text_content'), # or 'text_preview'
            #         **doc.get('metadata', {})
            #     })
            # return formatted_results

            logger.info(f"Retrieved {len(results)} documents for query '{query}'.")
            return results

        except Exception as e:
            logger.error(f"Error during knowledge base search for query '{query}': {e}", exc_info=True)
            return []

if __name__ == '__main__':
    # This is a placeholder for basic testing.
    # To run this, you'd need actual or mocked EmbeddingClient and VectorIndexer.
    # And a populated settings object.
    
    # Mock settings for local testing if needed
    class MockSettings:
        EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
        VECTOR_INDEX_PATH = "app/data/vector_store/index.faiss"
        EMBEDDING_DIMENSION = 10 # Example dimension
    
    # settings = MockSettings() # Uncomment and adjust if running standalone

    async def main():
        # Mock EmbeddingClient
        class MockEmbeddingClient(EmbeddingClient):
            async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
                logger.info(f"MockEmbeddingClient: Generating embeddings for {texts}")
                # Return a list of dummy embeddings, one for each text
                return [[0.1] * settings.EMBEDDING_DIMENSION for _ in texts]

        # Mock VectorIndexer
        class MockVectorIndexer(VectorIndexer):
            def __init__(self, index_path: str, embedding_dimension: int):
                super().__init__(index_path, embedding_dimension)
                self.dummy_data = {} # Store dummy data by ID
                self.is_loaded = False

            def load_index(self):
                logger.info("MockVectorIndexer: Attempting to load index.")
                # Simulate loading some data if index is "built"
                if self.index_path == "mock_index_built.faiss": # Simulate a built index
                    self.dummy_data = {
                        "doc1": {"original_id": "doc1", "text_preview": "This is content of doc1.", "source": "TestSource"},
                        "doc2": {"original_id": "doc2", "text_preview": "Content for document 2.", "source": "TestSource"},
                    }
                    self.is_loaded = True
                    logger.info("MockVectorIndexer: Simulated index loaded with 2 docs.")
                else:
                    logger.info("MockVectorIndexer: No data to load for this path.")
                    # raise FileNotFoundError("Mock index not found") # Or just stay empty

            def is_index_available(self) -> bool:
                return self.is_loaded
            
            def search(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
                logger.info(f"MockVectorIndexer: Searching with top_k={top_k}")
                if not self.is_loaded:
                    logger.warning("MockVectorIndexer: Search called but index not loaded.")
                    return []
                
                # Simulate returning some results based on dummy_data
                # This mock doesn't actually use the query_embedding
                results = []
                doc_ids = list(self.dummy_data.keys())
                for i in range(min(top_k, len(doc_ids))):
                    doc_id = doc_ids[i]
                    results.append({
                        "id": doc_id,
                        "score": 1.0 - (i * 0.1), # Dummy score
                        "text": self.dummy_data[doc_id]["text_preview"],
                        "metadata": {"source": self.dummy_data[doc_id]["source"]}
                    })
                return results

        # Test with mocks
        # Ensure settings are patched or provide mock settings for this test
        global settings # Allow modification for this test block
        
        original_settings_vector_path = settings.VECTOR_INDEX_PATH
        settings.VECTOR_INDEX_PATH = "mock_index_built.faiss" # Path that mock indexer "loads"

        mock_embed_client = MockEmbeddingClient(model_name=settings.EMBEDDING_MODEL_NAME)
        mock_vec_indexer = MockVectorIndexer(index_path=settings.VECTOR_INDEX_PATH, embedding_dimension=settings.EMBEDDING_DIMENSION)
        
        # Simulate building the index for the mock
        mock_vec_indexer.load_index() # This will populate dummy_data for the "built" path

        retrieval_service = RetrievalService(
            embedding_client=mock_embed_client,
            vector_indexer=mock_vec_indexer
        )

        test_query = "Tell me about finances."
        print(f"\nTesting search_knowledge_base with query: '{test_query}'")
        search_results = await retrieval_service.search_knowledge_base(test_query, top_k=3)
        
        if search_results:
            for i, res in enumerate(search_results):
                print(f"  Result {i+1}:")
                print(f"    ID: {res.get('id')}")
                print(f"    Score: {res.get('score')}")
                print(f"    Text: {res.get('text')}")
                print(f"    Metadata: {res.get('metadata')}")
        else:
            print("  No results found.")
        
        settings.VECTOR_INDEX_PATH = original_settings_vector_path # Restore

    if __name__ == '__main__':
        import asyncio
        logging.basicConfig(level=logging.INFO)
        # Ensure settings are loaded or mocked before running main
        # For example, by running from a context where app.core.config.settings is populated
        # If running this file directly, you might need to initialize settings explicitly or mock it.
        # Example:
        # from app.core.config import Settings
        # settings = Settings() # If your Settings class can be initialized like this
        asyncio.run(main())
