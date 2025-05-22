from __future__ import annotations

from pathlib import Path

from app.clients.embedding_client import EmbeddingClient
from app.rag_pipeline import (
    parse_train_json,
    chunk_documents,
    embed_chunks,
    build_index,
    save_index,
    save_metadata,
)


TRAIN_PATH = Path("data/train.json")
INDEX_PATH = Path("data/vector_store/faiss_index.idx")
METADATA_PATH = Path("data/vector_store/metadata_mapping.json")


def main() -> None:
    documents = parse_train_json(TRAIN_PATH)
    chunks = chunk_documents(documents)
    client = EmbeddingClient()
    embeddings, metadata = embed_chunks(chunks, client)
    index = build_index(embeddings)
    save_index(index, INDEX_PATH)
    save_metadata(metadata, METADATA_PATH)
    print("Vector store built at", INDEX_PATH)


if __name__ == "__main__":
    main()
