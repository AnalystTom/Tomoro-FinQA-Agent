# app/clients/embedding_client.py
import logging
from typing import List, Optional, Union

from tenacity import retry, stop_after_attempt, wait_random_exponential
import numpy as np

# Import settings from the application's core configuration
from app.config.settings import settings

# Conditional imports based on the client type
if settings.EMBEDDING_CLIENT_TYPE == "openai":
    try:
        from openai import OpenAI, APIError # Use the new OpenAI client
    except ImportError:
        logging.error("OpenAI library not installed. Please install it with 'pip install openai'")
        OpenAI = None # type: ignore 
        APIError = None # type: ignore
elif settings.EMBEDDING_CLIENT_TYPE == "local":
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logging.error("SentenceTransformers library not installed. Please install it with 'pip install sentence-transformers'")
        SentenceTransformer = None # type: ignore
else:
    logging.error(f"Unsupported EMBEDDING_CLIENT_TYPE: {settings.EMBEDDING_CLIENT_TYPE}")


class EmbeddingClient:
    def __init__(self):
        """
        Initializes the EmbeddingClient based on the configuration in settings.
        It can use either OpenAI or a local SentenceTransformer model.
        """
        self.client_type = settings.EMBEDDING_CLIENT_TYPE
        self.model_name = settings.EMBEDDING_MODEL_NAME
        self.client: Union[OpenAI, SentenceTransformer, None] = None # type: ignore

        if self.client_type == "openai":
            if not OpenAI or not APIError:
                logging.error("OpenAI client could not be initialized due to import error.")
                raise ImportError("OpenAI library not found or failed to import.")
            if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_HERE":
                logging.error("OpenAI API key is not configured. Please set OPENAI_API_KEY in your .env file or environment.")
                raise ValueError("OpenAI API key not configured.")
            try:
                self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
                logging.info(f"OpenAI EmbeddingClient initialized with model: {self.model_name}")
            except Exception as e:
                logging.error(f"Failed to initialize OpenAI client: {e}")
                raise
        elif self.client_type == "local":
            if not SentenceTransformer:
                logging.error("SentenceTransformer client could not be initialized due to import error.")
                raise ImportError("SentenceTransformers library not found or failed to import.")
            try:
                # Use SENTENCE_TRANSFORMERS_HOME if set, for caching models
                cache_folder = settings.SENTENCE_TRANSFORMERS_HOME if settings.SENTENCE_TRANSFORMERS_HOME else None
                self.client = SentenceTransformer(
                    model_name_or_path=self.model_name,
                    device=settings.LOCAL_EMBEDDING_DEVICE,
                    cache_folder=cache_folder
                )
                logging.info(f"Local SentenceTransformer EmbeddingClient initialized with model: {self.model_name} on device: {settings.LOCAL_EMBEDDING_DEVICE}")
            except Exception as e:
                logging.error(f"Failed to load local SentenceTransformer model '{self.model_name}': {e}")
                raise
        else:
            error_msg = f"Invalid EMBEDDING_CLIENT_TYPE configured: '{self.client_type}'. Must be 'openai' or 'local'."
            logging.error(error_msg)
            raise ValueError(error_msg)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _generate_openai_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a batch of texts using the OpenAI API."""
        if not isinstance(self.client, OpenAI): # Should be OpenAI type
             logging.error("OpenAI client not properly initialized for generating embeddings.")
             raise TypeError("OpenAI client is not initialized.")
        try:
            response = self.client.embeddings.create(input=texts, model=self.model_name)
            return [item.embedding for item in response.data]
        except APIError as e: # More specific error handling for OpenAI
            logging.error(f"OpenAI API error during embedding generation: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error during OpenAI embedding generation: {e}")
            raise

    def _generate_local_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a batch of texts using a local SentenceTransformer model."""
        if not isinstance(self.client, SentenceTransformer): # Should be SentenceTransformer type
            logging.error("SentenceTransformer client not properly initialized for generating embeddings.")
            raise TypeError("SentenceTransformer client is not initialized.")
        try:
            # The encode method returns a list of numpy arrays or a 2D numpy array.
            # Convert to list of lists of floats.
            embeddings_np = self.client.encode(texts, convert_to_numpy=True)
            return embeddings_np.tolist()
        except Exception as e:
            logging.error(f"Error generating local embeddings with SentenceTransformer: {e}")
            raise

    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generates embeddings for a list of texts, handling batching.
        Adjust batch_size based on the model and system memory.
        For OpenAI, API limits also apply (max tokens per request).
        For local models, batch_size affects memory usage and speed.
        """
        if self.client is None:
            logging.error("Embedding client is not initialized.")
            raise ValueError("Embedding client not initialized. Check configuration.")

        all_embeddings: List[List[float]] = []
        num_texts = len(texts)

        for i in range(0, num_texts, batch_size):
            batch = texts[i:i + batch_size]
            logging.debug(f"Generating embeddings for batch {i//batch_size + 1} (size: {len(batch)}) using {self.client_type} client.")
            
            try:
                if self.client_type == "openai":
                    batch_embeddings = self._generate_openai_embeddings_batch(batch)
                elif self.client_type == "local":
                    batch_embeddings = self._generate_local_embeddings_batch(batch)
                else:
                    # This case should ideally be caught in __init__
                    logging.error(f"Unsupported client type '{self.client_type}' in generate_embeddings.")
                    raise ValueError(f"Unsupported client type: {self.client_type}")
                
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logging.error(f"Failed to generate embeddings for batch starting at index {i} with {self.client_type} client: {e}")
                # Depending on desired behavior, you might want to collect partial results
                # or re-raise the exception to stop the whole process.
                # For now, re-raising to indicate failure of the overall operation.
                raise

        if len(all_embeddings) != num_texts:
            logging.warning(f"Number of generated embeddings ({len(all_embeddings)}) does not match number of input texts ({num_texts}).")
            # This might indicate an issue with batch processing or error handling within loops.

        return all_embeddings

