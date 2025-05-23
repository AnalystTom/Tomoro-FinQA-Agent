# scripts/ingest_data.py
import json
import os
import logging
from typing import List, Dict, Any

# Import your RAG pipeline components
from app.rag_pipeline.parser import DataParser
from app.rag_pipeline.chunker import Chunker
from app.clients.embedding_client import EmbeddingClient # Updated client
from app.rag_pipeline.indexer import VectorIndexer
from app.config.settings import settings # For API keys, model names, paths

# --- Configuration ---
logging.basicConfig(level=settings.LOG_LEVEL.upper(),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(), # Log to console
                        # logging.FileHandler(settings.LOG_FILE_PATH) # Optionally log to file
                    ])
logger = logging.getLogger(__name__) # It's good practice to use named loggers

# --- Initialization ---
logger.info("Initializing components...")
try:
    parser = DataParser()
    chunker = Chunker()
    # MODIFIED LINE: EmbeddingClient now takes no arguments in __init__
    # It reads configuration directly from 'settings'
    embedding_client = EmbeddingClient()
    vector_indexer = VectorIndexer(
        db_type=settings.VECTOR_DB_TYPE,
        store_path=settings.VECTOR_STORE_PATH,
        embedding_dimension=settings.EMBEDDING_DIMENSION
    )
    logger.info("All components initialized successfully.")
except Exception as e:
    logger.error(f"Error during component initialization: {e}", exc_info=True)
    # Depending on the severity, you might want to exit here
    raise # Re-raise the exception to stop execution if init fails

def main():
    logger.info("Starting Stage 1: Offline Data Preparation & Indexing.")

    # 1. Load and Iterate train.json
    logger.info(f"Loading data from {settings.TRAIN_JSON_PATH}...")
    try:
        with open(settings.TRAIN_JSON_PATH, 'r', encoding='utf-8') as f:
            raw_data_items = json.load(f)
        logger.info(f"Successfully loaded {len(raw_data_items)} items from {settings.TRAIN_JSON_PATH}.")
    except FileNotFoundError:
        logger.error(f"Error: {settings.TRAIN_JSON_PATH} not found.")
        return
    except json.JSONDecodeError:
        logger.error(f"Error: Could not decode JSON from {settings.TRAIN_JSON_PATH}.")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading data: {e}", exc_info=True)
        return

    all_chunks_for_indexing = []
    processed_item_count = 0
    failed_item_count = 0

    for i, item in enumerate(raw_data_items):
        item_id_for_log = item.get('id', f'Unknown ID at index {i}')
        logger.debug(f"Processing item {i+1}/{len(raw_data_items)} with ID: {item_id_for_log}")

        # 2. Content Extraction and Structuring
        try:
            parsed_content = parser.parse_item(item)
            if not parsed_content or "consolidated_text" not in parsed_content:
                logger.warning(f"Skipping item {item_id_for_log} due to parsing error or missing essential data.")
                failed_item_count +=1
                continue
        except Exception as e:
            logger.error(f"Error parsing item {item_id_for_log}: {e}", exc_info=True)
            failed_item_count +=1
            continue

        # 3. Chunking (Potentially trivial for ConvFinQA items)
        try:
            # Assuming chunk_document returns a list, and for ConvFinQA, it's a list with one item.
            document_chunks = chunker.chunk_document(parsed_content)
            if not document_chunks:
                logger.warning(f"No chunks produced for item {item_id_for_log}. Skipping.")
                failed_item_count +=1
                continue
            # We expect one chunk per item for this setup
            chunk_data = document_chunks[0]
            all_chunks_for_indexing.append(chunk_data)
            processed_item_count += 1
        except Exception as e:
            logger.error(f"Error chunking item {item_id_for_log}: {e}", exc_info=True)
            failed_item_count +=1
            continue


    if not all_chunks_for_indexing:
        logger.error(f"No data to process after parsing and chunking. Processed: {processed_item_count}, Failed: {failed_item_count}. Exiting.")
        return

    logger.info(f"Successfully parsed and prepared {processed_item_count} items for embedding. Failed items: {failed_item_count}.")

    # 4. Embedding
    logger.info(f"Generating embeddings for {len(all_chunks_for_indexing)} chunks using model '{settings.EMBEDDING_MODEL_NAME}' via '{settings.EMBEDDING_CLIENT_TYPE}' client...")
    texts_to_embed = [chunk["text_to_embed"] for chunk in all_chunks_for_indexing]
    
    embeddings = [] # Initialize embeddings list
    try:
        # Define a reasonable batch size, especially for local models
        batch_size_for_embedding = 32 if settings.EMBEDDING_CLIENT_TYPE == "local" else 512 # Larger for OpenAI if texts are small
        embeddings = embedding_client.generate_embeddings(texts_to_embed, batch_size=batch_size_for_embedding)
        logger.info(f"Successfully generated {len(embeddings)} embeddings.")
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}", exc_info=True)
        return # Stop if embeddings fail

    if len(embeddings) != len(all_chunks_for_indexing):
        logger.error(f"Mismatch between number of chunks ({len(all_chunks_for_indexing)}) and generated embeddings ({len(embeddings)}). Aborting indexing.")
        return

    # Combine chunks with their embeddings
    documents_to_index = []
    for i, chunk_data in enumerate(all_chunks_for_indexing):
        # Ensure metadata and original_id exist
        metadata = chunk_data.get("metadata", {})
        original_id = metadata.get("original_id", f"generated_id_{i}") # Fallback ID

        documents_to_index.append({
            "id": original_id,
            "text": chunk_data["text_to_embed"],
            "embedding": embeddings[i],
            "metadata": metadata
        })

    # 5. Indexing
    logger.info(f"Indexing {len(documents_to_index)} documents into vector store (type: {settings.VECTOR_DB_TYPE}) at {settings.VECTOR_STORE_PATH}...")
    try:
        vector_indexer.index_documents(documents_to_index)
        logger.info("Successfully indexed all documents.")
    except Exception as e:
        logger.error(f"Failed to index documents: {e}", exc_info=True)
        return

    logger.info("Stage 1: Offline Data Preparation & Indexing COMPLETED.")

if __name__ == "__main__":
    # This structure ensures that main() is called only when the script is executed directly
    try:
        main()
    except Exception as e:
        # Catch any unhandled exceptions from main and log them before exiting
        logger.critical(f"An unhandled exception occurred in main: {e}", exc_info=True)
        # Exit with a non-zero status code to indicate failure
        import sys
        sys.exit(1)
