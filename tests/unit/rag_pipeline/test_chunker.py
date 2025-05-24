import pytest
from app.rag_pipeline.chunker import Chunker, Chunk

class TestChunker:

    def test_chunker_initialization(self):
        chunker = Chunker()
        assert chunker.chunk_size is None
        assert chunker.chunk_overlap is None

        chunker_with_params = Chunker(chunk_size=512, chunk_overlap=50)
        assert chunker_with_params.chunk_size == 512
        assert chunker_with_params.chunk_overlap == 50

    def test_chunk_document_valid_input(self):
        chunker = Chunker()
        parsed_document = {
            "consolidated_text": "This is a test document content.",
            "metadata": {
                "original_id": "doc_123",
                "source": "test_source.pdf",
                "page": 1
            }
        }
        
        chunks = chunker.chunk_document(parsed_document)
        
        assert len(chunks) == 1
        chunk = chunks[0]
        assert isinstance(chunk, Chunk)
        assert chunk.document_id == "doc_123"
        assert chunk.text_to_embed == "This is a test document content."
        assert chunk.metadata == parsed_document["metadata"]
        assert chunk.id == "doc_123_chunk_0"

    def test_chunk_document_missing_consolidated_text(self):
        chunker = Chunker()
        parsed_document = {
            "metadata": {
                "original_id": "doc_456"
            }
        }
        
        chunks = chunker.chunk_document(parsed_document)
        
        assert len(chunks) == 0 # Should return an empty list

    def test_chunk_document_empty_metadata(self):
        chunker = Chunker()
        parsed_document = {
            "consolidated_text": "Content with empty metadata.",
            "metadata": {}
        }
        
        chunks = chunker.chunk_document(parsed_document)
        
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.document_id == "unknown_document_id"
        assert chunk.text_to_embed == "Content with empty metadata."
        assert chunk.metadata == {}
        assert chunk.id == "unknown_document_id_chunk_0"

    def test_chunk_document_missing_original_id_in_metadata(self):
        chunker = Chunker()
        parsed_document = {
            "consolidated_text": "Content with metadata but no original_id.",
            "metadata": {
                "source": "another_source.txt"
            }
        }
        
        chunks = chunker.chunk_document(parsed_document)
        
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.document_id == "unknown_document_id"
        assert chunk.text_to_embed == "Content with metadata but no original_id."
        assert chunk.metadata == parsed_document["metadata"]
        assert chunk.id == "unknown_document_id_chunk_0"

    def test_chunk_document_no_metadata_key(self):
        chunker = Chunker()
        parsed_document = {
            "consolidated_text": "Content with no metadata key."
        }
        
        chunks = chunker.chunk_document(parsed_document)
        
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.document_id == "unknown_document_id"
        assert chunk.text_to_embed == "Content with no metadata key."
        assert chunk.metadata == {} # Should default to empty dict
        assert chunk.id == "unknown_document_id_chunk_0"
