import pytest
import faiss
import numpy as np
import os
import json
import shutil # For cleaning up test directories

@pytest.fixture
def indexer_paths_and_dim(tmp_path):
    test_index_dir = tmp_path / "test_faiss_data"
    test_index_path = test_index_dir / "test_index.idx"
    test_metadata_path = test_index_dir / "test_metadata.json"
    embedding_dimension = 10
    return str(test_index_path), str(test_metadata_path), embedding_dimension, str(test_index_dir)

@pytest.fixture(autouse=True)
def mock_faiss_and_os(mocker, indexer_paths_and_dim):
    test_index_path, test_metadata_path, _, test_index_dir = indexer_paths_and_dim

    # Ensure test directory is clean before each test
    if os.path.exists(test_index_dir):
        shutil.rmtree(test_index_dir)
    os.makedirs(test_index_dir, exist_ok=True)

    # Patch os.makedirs to prevent actual directory creation during init tests
    mocker.patch('os.makedirs')
    
    # Patch os.path.exists for load/save tests
    mock_exists = mocker.patch('os.path.exists')
    mock_exists.return_value = False # Default: no files exist

    # Patch faiss.write_index and faiss.read_index
    mock_faiss_write = mocker.patch('faiss.write_index')
    mock_faiss_read = mocker.patch('faiss.read_index')

    # Patch builtins.open for metadata file operations
    mock_open = mocker.patch('builtins.open', mock_open())

    yield {
        "mock_exists": mock_exists,
        "mock_faiss_write": mock_faiss_write,
        "mock_faiss_read": mock_faiss_read,
        "mock_open": mock_open,
        "test_index_path": test_index_path,
        "test_metadata_path": test_metadata_path,
        "test_index_dir": test_index_dir
    }

class TestVectorIndexer:

    def test_initialization_no_existing_files(self, indexer_paths_and_dim, mock_faiss_and_os, mocker):
        test_index_path, test_metadata_path, embedding_dimension, test_index_dir = indexer_paths_and_dim
        
        # Mock _initialize_faiss_index to ensure it's called
        mock_init_faiss = mocker.patch.object(VectorIndexer, '_initialize_faiss_index')
        
        indexer = VectorIndexer(test_index_path, test_metadata_path, embedding_dimension)
        
        assert indexer is not None
        assert indexer.index_path == test_index_path
        assert indexer.metadata_path == test_metadata_path
        assert indexer.embedding_dimension == embedding_dimension
        assert indexer.next_id == 0
        assert indexer.metadata == {}
        mock_faiss_and_os["mock_makedirs"].assert_called_with(os.path.dirname(test_index_path), exist_ok=True)
        mock_faiss_and_os["mock_makedirs"].assert_called_with(os.path.dirname(test_metadata_path), exist_ok=True)
        mock_init_faiss.assert_called_once() # Called because no index file exists
        mock_faiss_and_os["mock_faiss_read"].assert_not_called()
        mock_faiss_and_os["mock_open"].assert_not_called() # No metadata file to open

    def test_initialization_existing_files_loaded_successfully(self, indexer_paths_and_dim, mock_faiss_and_os, mocker):
        test_index_path, test_metadata_path, embedding_dimension, _ = indexer_paths_and_dim
        mock_faiss_and_os["mock_exists"].side_effect = lambda path: path in [test_index_path, test_metadata_path]
        
        mock_faiss_index = MagicMock(spec=faiss.IndexIDMap)
        mock_faiss_index.ntotal = 5
        mock_faiss_and_os["mock_faiss_read"].return_value = mock_faiss_index

        mock_metadata_content = json.dumps({"metadata": {"0": {"id": "doc0"}, "1": {"id": "doc1"}}, "next_id": 2})
        mock_faiss_and_os["mock_open"].return_value.__enter__.return_value.read.return_value = mock_metadata_content

        indexer = VectorIndexer(test_index_path, test_metadata_path, embedding_dimension)
        
        assert indexer.index is mock_faiss_index
        assert indexer.metadata == {0: {"id": "doc0"}, 1: {"id": "doc1"}}
        assert indexer.next_id == 2
        mock_faiss_and_os["mock_faiss_read"].assert_called_once_with(test_index_path)
        mock_faiss_and_os["mock_open"].assert_called_once_with(test_metadata_path, 'r')

    def test_initialization_index_load_failure(self, indexer_paths_and_dim, mock_faiss_and_os, mocker):
        test_index_path, test_metadata_path, embedding_dimension, _ = indexer_paths_and_dim
        mock_faiss_and_os["mock_exists"].side_effect = lambda path: path == test_index_path # Only index file exists
        mock_faiss_and_os["mock_faiss_read"].side_effect = RuntimeError("Corrupt index")
        
        mock_init_faiss = mocker.patch.object(VectorIndexer, '_initialize_faiss_index')
        
        indexer = VectorIndexer(test_index_path, test_metadata_path, embedding_dimension)
        assert indexer.index is None # Should be None if _initialize_faiss_index fails to set it
        mock_init_faiss.assert_called_once() # Should attempt to re-initialize
        mock_faiss_and_os["mock_faiss_read"].assert_called_once_with(test_index_path)
        assert indexer.next_id == 0 # Reset if metadata not loaded or index fails

    def test_initialization_metadata_load_failure(self, indexer_paths_and_dim, mock_faiss_and_os, mocker):
        test_index_path, test_metadata_path, embedding_dimension, _ = indexer_paths_and_dim
        mock_faiss_and_os["mock_exists"].side_effect = lambda path: path == test_metadata_path # Only metadata file exists
        mock_faiss_and_os["mock_open"].return_value.__enter__.return_value.read.side_effect = json.JSONDecodeError("Corrupt JSON", "", 0)
        
        mock_init_faiss = mocker.patch.object(VectorIndexer, '_initialize_faiss_index')
        
        indexer = VectorIndexer(test_index_path, test_metadata_path, embedding_dimension)
        assert indexer.metadata == {} # Should be empty
        assert indexer.next_id == 0 # Should be reset
        mock_init_faiss.assert_called_once() # Still initializes a new FAISS index
        mock_faiss_and_os["mock_open"].assert_called_once_with(test_metadata_path, 'r')

    def test_add_documents_success(self, indexer_paths_and_dim, mocker):
        test_index_path, test_metadata_path, embedding_dimension, _ = indexer_paths_and_dim
        indexer = VectorIndexer(test_index_path, test_metadata_path, embedding_dimension)
        indexer._initialize_faiss_index() # Manually initialize for testing add_documents
        
        mock_add_with_ids = mocker.MagicMock()
        indexer.index.add_with_ids = mock_add_with_ids # Mock the add_with_ids method

        embeddings = np.random.rand(2, embedding_dimension).astype('float32')
        metadata = [{"original_id": "docA", "text_preview": "textA"}, {"original_id": "docB", "text_preview": "textB"}]
        
        indexer.add_documents(embeddings, metadata)
        
        mock_add_with_ids.assert_called_once()
        assert indexer.index.ntotal == 2
        assert indexer.next_id == 2
        assert indexer.metadata[0]["original_id"] == "docA"
        assert indexer.metadata[1]["original_id"] == "docB"

    def test_add_documents_initial_index_none(self, indexer_paths_and_dim, mocker):
        # Test that add_documents calls _initialize_faiss_index if index is None
        test_index_path, test_metadata_path, embedding_dimension, _ = indexer_paths_and_dim
        indexer = VectorIndexer(test_index_path, test_metadata_path, embedding_dimension)
        indexer.index = None # Ensure it's None
        
        mock_init_faiss = mocker.patch.object(indexer, '_initialize_faiss_index')
        mock_faiss_index = mocker.MagicMock(spec=faiss.IndexIDMap)
        mock_faiss_index.add_with_ids = mocker.MagicMock()
        mock_faiss_index.ntotal = 0 # Initial state
        mock_init_faiss.side_effect = lambda: setattr(indexer, 'index', mock_faiss_index)

        embeddings = np.random.rand(1, embedding_dimension).astype('float32')
        metadata = [{"original_id": "docX", "text_preview": "textX"}]
        
        indexer.add_documents(embeddings, metadata)
        
        mock_init_faiss.assert_called_once()
        mock_faiss_index.add_with_ids.assert_called_once()
        assert indexer.next_id == 1

    def test_add_documents_invalid_embedding_dimension(self, indexer_paths_and_dim):
        test_index_path, test_metadata_path, embedding_dimension, _ = indexer_paths_and_dim
        indexer = VectorIndexer(test_index_path, test_metadata_path, embedding_dimension)
        indexer._initialize_faiss_index()
        
        embeddings = np.random.rand(1, embedding_dimension + 1).astype('float32') # Wrong dimension
        metadata = [{"original_id": "docA", "text_preview": "textA"}]
        
        indexer.add_documents(embeddings, metadata)
        assert indexer.index.ntotal == 0 # No documents added
        assert indexer.next_id == 0
        assert indexer.metadata == {}

    def test_add_documents_mismatched_counts(self, indexer_paths_and_dim):
        test_index_path, test_metadata_path, embedding_dimension, _ = indexer_paths_and_dim
        indexer = VectorIndexer(test_index_path, test_metadata_path, embedding_dimension)
        indexer._initialize_faiss_index()
        
        embeddings = np.random.rand(2, embedding_dimension).astype('float32')
        metadata = [{"original_id": "docA", "text_preview": "textA"}] # Only one metadata entry
        
        indexer.add_documents(embeddings, metadata)
        assert indexer.index.ntotal == 0 # No documents added
        assert indexer.next_id == 0
        assert indexer.metadata == {}

    def test_search_success(self, indexer_paths_and_dim, mocker):
        test_index_path, test_metadata_path, embedding_dimension, _ = indexer_paths_and_dim
        indexer = VectorIndexer(test_index_path, test_metadata_path, embedding_dimension)
        indexer._initialize_faiss_index()
        
        # Populate index with dummy data
        embeddings = np.array([
            [0.1]*embedding_dimension,
            [0.9]*embedding_dimension
        ], dtype=np.float32)
        metadata = [
            {"original_id": "doc1", "text_preview": "Content of doc1"},
            {"original_id": "doc2", "text_preview": "Content of doc2"}
        ]
        indexer.add_documents(embeddings, metadata)

        # Mock FAISS search result
        mock_distances = np.array([[0.0, 1.0]], dtype=np.float32) # L2 distance
        mock_faiss_ids = np.array([[0, 1]], dtype='int64')
        indexer.index.search = mocker.MagicMock(return_value=(mock_distances, mock_faiss_ids))

        query_embedding = np.array([0.1]*embedding_dimension).astype('float32') # Similar to doc1
        results = indexer.search(query_embedding, top_k=2)
        
        assert len(results) == 2
        assert results[0]["id"] == "doc1"
        assert results[0]["score"] == 0.0 # Raw L2 distance
        assert results[0]["text"] == "Content of doc1"
        assert results[1]["id"] == "doc2"
        assert results[1]["score"] == 1.0
        assert results[1]["text"] == "Content of doc2"
        indexer.index.search.assert_called_once()

    def test_search_empty_index(self, indexer_paths_and_dim):
        test_index_path, test_metadata_path, embedding_dimension, _ = indexer_paths_and_dim
        indexer = VectorIndexer(test_index_path, test_metadata_path, embedding_dimension)
        # Index is empty by default after init if no files exist
        query_embedding = np.random.rand(embedding_dimension).astype('float32')
        results = indexer.search(query_embedding)
        assert results == []

    def test_search_invalid_query_embedding_shape(self, indexer_paths_and_dim):
        test_index_path, test_metadata_path, embedding_dimension, _ = indexer_paths_and_dim
        indexer = VectorIndexer(test_index_path, test_metadata_path, embedding_dimension)
        indexer._initialize_faiss_index()
        
        # 3D array
        query_embedding = np.random.rand(1, 1, embedding_dimension).astype('float32')
        results = indexer.search(query_embedding)
        assert results == []

        # Wrong dimension
        query_embedding = np.random.rand(embedding_dimension + 1).astype('float32')
        results = indexer.search(query_embedding)
        assert results == []

    def test_search_faiss_id_minus_one(self, indexer_paths_and_dim, mocker):
        test_index_path, test_metadata_path, embedding_dimension, _ = indexer_paths_and_dim
        indexer = VectorIndexer(test_index_path, test_metadata_path, embedding_dimension)
        indexer._initialize_faiss_index()
        
        embeddings = np.array([[0.1]*embedding_dimension], dtype=np.float32)
        metadata = [{"original_id": "doc1", "text_preview": "Content of doc1"}]
        indexer.add_documents(embeddings, metadata)

        # Simulate FAISS returning -1 for one of the results (padding or no match)
        mock_distances = np.array([[0.0, 999.0]], dtype=np.float32)
        mock_faiss_ids = np.array([[0, -1]], dtype='int64')
        indexer.index.search = mocker.MagicMock(return_value=(mock_distances, mock_faiss_ids))

        query_embedding = np.array([0.1]*embedding_dimension).astype('float32')
        results = indexer.search(query_embedding, top_k=2)
        
        assert len(results) == 1 # Only doc0 should be returned
        assert results[0]["id"] == "doc1"

    def test_save_index_and_metadata_success(self, indexer_paths_and_dim, mock_faiss_and_os):
        test_index_path, test_metadata_path, embedding_dimension, _ = indexer_paths_and_dim
        indexer = VectorIndexer(test_index_path, test_metadata_path, embedding_dimension)
        indexer._initialize_faiss_index()
        indexer.index.ntotal = 1 # Simulate some data
        indexer.metadata = {0: {"original_id": "doc0"}}
        indexer.next_id = 1

        indexer.save_index_and_metadata()
        
        mock_faiss_and_os["mock_faiss_write"].assert_called_once_with(indexer.index, test_index_path)
        mock_faiss_and_os["mock_open"].assert_called_once_with(test_metadata_path, 'w')
        # Verify content written to metadata file
        written_content = mock_faiss_and_os["mock_open"].return_value.__enter__.return_value.write.call_args[0][0]
        loaded_data = json.loads(written_content)
        assert loaded_data["next_id"] == 1
        assert loaded_data["metadata"] == {"0": {"original_id": "doc0"}}

    def test_save_index_and_metadata_no_index(self, indexer_paths_and_dim, mock_faiss_and_os):
        test_index_path, test_metadata_path, embedding_dimension, _ = indexer_paths_and_dim
        indexer = VectorIndexer(test_index_path, test_metadata_path, embedding_dimension)
        indexer.index = None # Simulate no index
        indexer.metadata = {0: {"original_id": "doc0"}}
        indexer.next_id = 1

        indexer.save_index_and_metadata()
        
        mock_faiss_and_os["mock_faiss_write"].assert_not_called() # Should not try to write FAISS index
        mock_faiss_and_os["mock_open"].assert_called_once_with(test_metadata_path, 'w') # Should still try to write metadata

    def test_load_index_and_metadata_consistency_next_id(self, indexer_paths_and_dim, mock_faiss_and_os):
        test_index_path, test_metadata_path, embedding_dimension, _ = indexer_paths_and_dim
        mock_faiss_and_os["mock_exists"].side_effect = lambda path: path in [test_index_path, test_metadata_path]
        
        mock_faiss_index = MagicMock(spec=faiss.IndexIDMap)
        mock_faiss_index.ntotal = 5 # Index has 5 items
        mock_faiss_and_os["mock_faiss_read"].return_value = mock_faiss_index

        # Metadata has max ID 1, next_id 0 (inconsistent)
        mock_metadata_content = json.dumps({"metadata": {"0": {"id": "doc0"}, "1": {"id": "doc1"}}, "next_id": 0})
        mock_faiss_and_os["mock_open"].return_value.__enter__.return_value.read.return_value = mock_metadata_content

        indexer = VectorIndexer(test_index_path, test_metadata_path, embedding_dimension)
        
        # next_id should be adjusted to max_stored_id + 1
        assert indexer.next_id == 2 # max(0,1) + 1 = 2

    def test_is_index_available(self, indexer_paths_and_dim):
        test_index_path, test_metadata_path, embedding_dimension, _ = indexer_paths_and_dim
        indexer = VectorIndexer(test_index_path, test_metadata_path, embedding_dimension)
        assert not indexer.is_index_available() # Initially not available

        indexer._initialize_faiss_index()
        assert not indexer.is_index_available() # Available but ntotal is 0

        indexer.index.ntotal = 1 # Simulate adding one document
        assert indexer.is_index_available()

    def test_clear_index(self, indexer_paths_and_dim):
        test_index_path, test_metadata_path, embedding_dimension, _ = indexer_paths_and_dim
        indexer = VectorIndexer(test_index_path, test_metadata_path, embedding_dimension)
        indexer._initialize_faiss_index()
        indexer.add_documents(np.random.rand(1, embedding_dimension).astype('float32'), [{"original_id": "doc1"}])
        
        assert indexer.is_index_available()
        assert indexer.next_id == 1
        assert not not indexer.metadata

        indexer.clear_index()
        
        assert not indexer.is_index_available()
        assert indexer.next_id == 0
        assert indexer.metadata == {}
        assert indexer.index is not None # Index object itself should still exist, just empty
