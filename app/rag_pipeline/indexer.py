# app/rag_pipeline/indexer.py
import faiss
import numpy as np
import os
import json
import logging
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

class VectorIndexer:
    """
    Handles the creation, loading, and searching of a FAISS vector index.
    Also manages associated metadata.
    """
    def __init__(self, index_path: str, metadata_path: str, embedding_dimension: int):
        """
        Initializes the VectorIndexer.

        Args:
            index_path: Path to save/load the FAISS index file.
            metadata_path: Path to save/load the metadata JSON file.
            embedding_dimension: The dimension of the embeddings.
        """
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embedding_dimension = embedding_dimension
        self.index: Optional[faiss.IndexIDMap] = None
        self.metadata: Dict[int, Dict[str, Any]] = {} # Maps FAISS index ID to document metadata
        self.next_id = 0 # To assign unique IDs for FAISS if adding incrementally

        # Ensure directory for index and metadata exists
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)

        # Try to load existing index and metadata
        self.load_index_and_metadata()

    def _initialize_faiss_index(self):
        """Initializes a new FAISS index."""
        # Using IndexFlatL2 for simplicity, but IndexIVFFlat or others could be used for larger datasets.
        # IndexIDMap allows us to use our own document IDs.
        try:
            inner_index = faiss.IndexFlatL2(self.embedding_dimension)
            self.index = faiss.IndexIDMap(inner_index)
            self.next_id = 0 # Reset next_id for a new index
            logger.info(f"Initialized new FAISS IndexIDMap with L2 metric, dimension {self.embedding_dimension}.")
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}", exc_info=True)
            self.index = None


    def add_documents(self, embeddings: np.ndarray, documents_metadata: List[Dict[str, Any]]):
        """
        Adds document embeddings and their metadata to the index.

        Args:
            embeddings: A NumPy array of document embeddings (num_documents, embedding_dimension).
            documents_metadata: A list of metadata dictionaries, one for each document.
                                Each dict should at least contain 'original_id' and 'text_preview'.
        """
        if self.index is None:
            self._initialize_faiss_index()
            if self.index is None: # Still None after attempt to initialize
                logger.error("Cannot add documents: FAISS index is not initialized.")
                return

        if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
            logger.error("Embeddings must be a 2D NumPy array.")
            return
        if embeddings.shape[1] != self.embedding_dimension:
            logger.error(f"Embedding dimension mismatch. Expected {self.embedding_dimension}, got {embeddings.shape[1]}.")
            return
        if embeddings.shape[0] != len(documents_metadata):
            logger.error("Number of embeddings does not match number of metadata entries.")
            return

        num_documents = embeddings.shape[0]
        ids_to_add = np.arange(self.next_id, self.next_id + num_documents).astype('int64')

        try:
            self.index.add_with_ids(embeddings.astype('float32'), ids_to_add)
            for i, doc_meta in enumerate(documents_metadata):
                current_faiss_id = ids_to_add[i]
                self.metadata[int(current_faiss_id)] = doc_meta # Store metadata
            self.next_id += num_documents
            logger.info(f"Added {num_documents} documents to the index. Total documents: {self.index.ntotal}")
        except Exception as e:
            logger.error(f"Error adding documents to FAISS index: {e}", exc_info=True)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches the index for the most similar documents to the query embedding.

        Args:
            query_embedding: A 1D NumPy array representing the query embedding.
            top_k: The number of top results to retrieve.

        Returns:
            A list of dictionaries, where each dictionary contains the document's
            metadata and its similarity score.
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is not initialized or is empty. Cannot perform search.")
            return []
        
        if not isinstance(query_embedding, np.ndarray):
            logger.error("Query embedding must be a NumPy array.")
            return []

        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0).astype('float32')
        elif query_embedding.ndim == 2 and query_embedding.shape[0] == 1:
            query_embedding = query_embedding.astype('float32')
        else:
            logger.error(f"Query embedding has incorrect shape: {query_embedding.shape}. Expected (dim,) or (1, dim).")
            return []

        if query_embedding.shape[1] != self.embedding_dimension:
            logger.error(f"Query embedding dimension mismatch. Expected {self.embedding_dimension}, got {query_embedding.shape[1]}.")
            return []

        try:
            distances, faiss_ids = self.index.search(query_embedding, top_k)
            results = []
            for i in range(faiss_ids.shape[1]):
                faiss_id = int(faiss_ids[0, i])
                if faiss_id == -1: # FAISS uses -1 for no result / padding
                    continue 
                
                doc_metadata = self.metadata.get(faiss_id)
                if doc_metadata:
                    # L2 distance, smaller is better. Convert to a similarity score (e.g., 1 / (1 + distance))
                    # Or if using cosine similarity index (IndexFlatIP), distances are dot products (larger is better).
                    # For IndexFlatL2, distances are squared L2.
                    score = float(distances[0, i]) # Raw L2 distance, or 1.0 - distance for normalized vectors & IP
                    results.append({
                        "id": doc_metadata.get("original_id", f"faiss_id_{faiss_id}"),
                        "score": score, # Note: This is distance, lower is better for L2.
                                        # The tool layer might want to invert this for "similarity".
                        "text": doc_metadata.get("text_preview", "N/A"), # Ensure 'text_preview' or 'text' is in metadata
                        "metadata": doc_metadata # Full metadata for potential further use
                    })
            logger.info(f"Search completed. Found {len(results)} relevant documents for top_k={top_k}.")
            return results
        except Exception as e:
            logger.error(f"Error during FAISS search: {e}", exc_info=True)
            return []

    def save_index_and_metadata(self):
        """Saves the FAISS index and metadata to disk."""
        if self.index:
            try:
                faiss.write_index(self.index, self.index_path)
                logger.info(f"FAISS index saved to {self.index_path}")
            except Exception as e:
                logger.error(f"Error saving FAISS index: {e}", exc_info=True)
        
        try:
            with open(self.metadata_path, 'w') as f:
                # Convert int keys in self.metadata to str for JSON compatibility
                json_compatible_metadata = {str(k): v for k, v in self.metadata.items()}
                json.dump({"metadata": json_compatible_metadata, "next_id": self.next_id}, f, indent=4)
            logger.info(f"Metadata saved to {self.metadata_path}")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}", exc_info=True)

    def load_index_and_metadata(self):
        """Loads the FAISS index and metadata from disk."""
        loaded_index_successfully = False
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
                logger.info(f"FAISS index loaded from {self.index_path}. Total documents: {self.index.ntotal if self.index else 'N/A'}")
                loaded_index_successfully = True
            except Exception as e:
                logger.warning(f"Could not load FAISS index from {self.index_path}: {e}. Will initialize a new one if needed.", exc_info=True)
                self._initialize_faiss_index() # Initialize a fresh index on load failure
        else:
            logger.info(f"No FAISS index file found at {self.index_path}. Will initialize a new one if documents are added.")
            self._initialize_faiss_index() # Initialize fresh if no file

        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r') as f:
                    data = json.load(f)
                    # Convert str keys back to int for self.metadata
                    self.metadata = {int(k): v for k, v in data.get("metadata", {}).items()}
                    self.next_id = data.get("next_id", 0) # Load next_id
                logger.info(f"Metadata loaded from {self.metadata_path}. {len(self.metadata)} entries. Next ID: {self.next_id}")
                # If index was loaded, ensure next_id is consistent or warn.
                # A simple consistency check: if index has items, next_id should reflect that.
                # For IndexIDMap, ntotal reflects added items, but IDs can be sparse.
                # self.next_id should ideally be max(existing_ids) + 1 or count of items if IDs are sequential.
                # The current self.next_id logic assumes sequential IDs starting from 0.
                if loaded_index_successfully and self.index and self.index.ntotal > 0:
                    if not self.metadata and self.index.ntotal > 0:
                         logger.warning("Index loaded with items, but metadata is empty. Metadata might be corrupt or missing.")
                    # If next_id from metadata seems too small given index.ntotal, it might indicate an issue.
                    # However, with IndexIDMap, IDs might not be dense from 0 to ntotal-1.
                    # A more robust check would be to ensure self.next_id is greater than any ID in self.metadata.keys()
                    if self.metadata:
                        max_stored_id = max(self.metadata.keys(), default=-1)
                        if self.next_id <= max_stored_id:
                            logger.warning(f"Loaded next_id ({self.next_id}) is not greater than max stored metadata ID ({max_stored_id}). Adjusting next_id.")
                            self.next_id = max_stored_id + 1


            except Exception as e:
                logger.warning(f"Could not load metadata from {self.metadata_path}: {e}. Initializing empty metadata.", exc_info=True)
                self.metadata = {}
                self.next_id = 0 # Reset if metadata load fails
        else:
            logger.info(f"No metadata file found at {self.metadata_path}. Initializing empty metadata.")
            self.metadata = {}
            self.next_id = 0
        
        # If index loaded but metadata didn't, or vice-versa, it's a potential inconsistency.
        # For now, we proceed, but a more robust system might try to rebuild/reconcile.


    def is_index_available(self) -> bool:
        """
        Checks if the FAISS index is initialized and contains data.
        This is the method your tests are looking for.
        """
        return self.index is not None and self.index.ntotal > 0

    def get_document_by_faiss_id(self, faiss_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves document metadata by its internal FAISS ID."""
        return self.metadata.get(faiss_id)

    def get_all_metadata(self) -> Dict[int, Dict[str, Any]]:
        """Returns all stored metadata."""
        return self.metadata

    def clear_index(self):
        """Clears the index and metadata."""
        self._initialize_faiss_index() # Re-initialize to an empty index
        self.metadata = {}
        self.next_id = 0
        logger.info("Vector index and metadata have been cleared.")
        # Optionally, delete the files from disk
        # if os.path.exists(self.index_path): os.remove(self.index_path)
        # if os.path.exists(self.metadata_path): os.remove(self.metadata_path)

# Example usage (for testing this file directly):
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Dummy settings for direct test
    DUMMY_INDEX_PATH = "temp_data/test_faiss_index.idx"
    DUMMY_METADATA_PATH = "temp_data/test_metadata.json"
    DUMMY_EMBED_DIM = 5 # Small dimension for testing

    # Clean up old files if they exist
    if os.path.exists(DUMMY_INDEX_PATH): os.remove(DUMMY_INDEX_PATH)
    if os.path.exists(DUMMY_METADATA_PATH): os.remove(DUMMY_METADATA_PATH)
    os.makedirs(os.path.dirname(DUMMY_INDEX_PATH), exist_ok=True)


    indexer = VectorIndexer(
        index_path=DUMMY_INDEX_PATH,
        metadata_path=DUMMY_METADATA_PATH,
        embedding_dimension=DUMMY_EMBED_DIM
    )

    print(f"Index available after init: {indexer.is_index_available()}") # Should be False

    # Add some documents
    docs_meta = [
        {"original_id": "doc1", "text_preview": "This is the first document about apples."},
        {"original_id": "doc2", "text_preview": "The second document discusses bananas."},
        {"original_id": "doc3", "text_preview": "Finally, a document on cherries."},
    ]
    # Create dummy embeddings (ensure they are float32 for FAISS)
    embeddings_np = np.array([
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.6, 0.7, 0.8, 0.9, 1.0],
        [1.1, 1.2, 1.3, 1.4, 1.5],
    ], dtype=np.float32)

    indexer.add_documents(embeddings_np, docs_meta)
    print(f"Index available after adding docs: {indexer.is_index_available()}") # Should be True
    print(f"Index total: {indexer.index.ntotal if indexer.index else 'N/A'}")

    # Save and clear
    indexer.save_index_and_metadata()
    indexer.clear_index()
    print(f"Index available after clear: {indexer.is_index_available()}") # Should be False
    print(f"Metadata after clear: {indexer.metadata}")


    # Load again
    print("Loading indexer again...")
    indexer_loaded = VectorIndexer(
        index_path=DUMMY_INDEX_PATH,
        metadata_path=DUMMY_METADATA_PATH,
        embedding_dimension=DUMMY_EMBED_DIM
    )
    print(f"Index available after loading: {indexer_loaded.is_index_available()}") # Should be True
    print(f"Index total after loading: {indexer_loaded.index.ntotal if indexer_loaded.index else 'N/A'}")
    print(f"Metadata after loading: {indexer_loaded.metadata}")


    # Search
    if indexer_loaded.is_index_available():
        query_emb = np.array([[0.15, 0.25, 0.35, 0.45, 0.55]], dtype=np.float32) # Similar to doc1
        search_results = indexer_loaded.search(query_emb, top_k=2)
        print("\nSearch Results:")
        for res in search_results:
            print(f"  ID: {res['id']}, Score (L2 dist): {res['score']:.4f}, Text: {res['text']}")
    
    # Clean up dummy files
    if os.path.exists(DUMMY_INDEX_PATH): os.remove(DUMMY_INDEX_PATH)
    if os.path.exists(DUMMY_METADATA_PATH): os.remove(DUMMY_METADATA_PATH)
    if os.path.exists("temp_data") and not os.listdir("temp_data"):
        os.rmdir("temp_data")
