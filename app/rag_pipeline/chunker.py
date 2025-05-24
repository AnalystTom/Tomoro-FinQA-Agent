# app/rag_pipeline/chunker.py
import logging
from typing import Dict, List, Any, Optional
from pydantic import BaseModel # Using Pydantic for data validation and structure

logger = logging.getLogger(__name__)

class Chunk(BaseModel):
    """
    Represents a chunk of text to be embedded, along with its metadata.
    """
    id: Optional[str] = None # Optional unique ID for the chunk itself
    document_id: Optional[str] = None # ID of the source document
    text_to_embed: str
    metadata: Dict[str, Any] = {}


class Chunker:
    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None): # Parameters might not be used for this strategy
        logger.info("Chunker initialized.")
        # For ConvFinQA, we might not need complex chunking logic as each item is a "natural" chunk.
        # These parameters are placeholders if a more complex strategy is needed later.
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, parsed_document: Dict[str, Any]) -> List[Chunk]:
        """
        For ConvFinQA, typically the parsed_document (consolidated_text + metadata)
        is treated as a single chunk. This method wraps it into a Chunk object.

        Args:
            parsed_document: A dictionary expected to contain at least
                             "consolidated_text" and "metadata" (which should
                             include an 'original_id' for the document).

        Returns:
            A list containing a single Chunk object.
        """
        consolidated_text = parsed_document.get("consolidated_text")
        metadata = parsed_document.get("metadata", {})
        document_id = metadata.get("original_id", "unknown_document_id") # Get original_id for document_id

        if consolidated_text is None:
            logger.warning("parsed_document missing 'consolidated_text'. Returning empty chunk list.")
            return []

        # Treat the entire parsed document as one chunk for this strategy
        # You might generate a unique chunk ID if needed, e.g., using hashlib or uuid
        chunk_id = f"{document_id}_chunk_0" 

        chunk = Chunk(
            id=chunk_id,
            document_id=document_id,
            text_to_embed=consolidated_text,
            metadata=metadata 
        )
        logger.debug(f"Created chunk for document_id: {document_id}")
        return [chunk]

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    
    # Example Usage
    chunker_instance = Chunker()
    
    sample_parsed_doc_1 = {
        "consolidated_text": "This is the first document. It contains financial data about company X.",
        "metadata": {
            "original_id": "doc_xyz_001",
            "source_filename": "report_q1.pdf",
            "page_number": 5
        }
    }
    
    sample_parsed_doc_2 = {
        "consolidated_text": "Another document discussing market trends.",
        "metadata": { # Metadata might be simpler or different for other docs
            "original_id": "doc_abc_002",
            "source_url": "http://example.com/trends"
        }
    }
    
    chunks1 = chunker_instance.chunk_document(sample_parsed_doc_1)
    if chunks1:
        print(f"\nChunks for doc_xyz_001 (ID: {chunks1[0].id}):")
        print(f"  Text: '{chunks1[0].text_to_embed[:50]}...'")
        print(f"  Metadata: {chunks1[0].metadata}")
    
    chunks2 = chunker_instance.chunk_document(sample_parsed_doc_2)
    if chunks2:
        print(f"\nChunks for doc_abc_002 (ID: {chunks2[0].id}):")
        print(f"  Text: '{chunks2[0].text_to_embed[:50]}...'")
        print(f"  Metadata: {chunks2[0].metadata}")

    # Example with missing consolidated_text
    sample_parsed_doc_3_missing_text = {
        "metadata": {
            "original_id": "doc_err_003"
        }
    }
    chunks3 = chunker_instance.chunk_document(sample_parsed_doc_3_missing_text)
    print(f"\nChunks for doc_err_003 (should be empty): {chunks3}")

