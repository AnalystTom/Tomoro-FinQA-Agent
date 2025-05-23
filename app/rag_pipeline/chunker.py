# app/rag_pipeline/chunker.py
import logging
from typing import Dict, List

class Chunker:
    def __init__(self, chunk_size=None, chunk_overlap=None): # Parameters might not be used for this strategy
        logging.info("Chunker initialized.")
        # For ConvFinQA, we might not need complex chunking logic as each item is a "natural" chunk.
        # These parameters are placeholders if a more complex strategy is needed later.
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, parsed_document: Dict) -> List[Dict]:
        """
        For ConvFinQA, typically the parsed_document (consolidated_text + metadata)
        is treated as a single chunk.
        If pre_text or post_text were extremely long, one might consider splitting them
        while keeping them linked to the table_text. But for now, one item = one chunk.
        """
        # The "chunk" is the entire consolidated text from the parser.
        # The metadata is associated with this single chunk.
        return [
            {
                "text_to_embed": parsed_document["consolidated_text"],
                "metadata": parsed_document["metadata"] # Pass along metadata from parser
            }
        ]