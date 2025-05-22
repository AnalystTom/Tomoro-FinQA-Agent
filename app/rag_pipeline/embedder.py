from __future__ import annotations

from typing import List, Tuple, Iterable
import numpy as np

from app.clients.embedding_client import EmbeddingClient
from .chunker import Chunk


def embed_chunks(chunks: Iterable[Chunk], client: EmbeddingClient) -> Tuple[np.ndarray, List[dict]]:
    texts = [c.text for c in chunks]
    embeddings = client.embed(texts)
    metadata = []
    for idx, c in enumerate(chunks):
        metadata.append(
            {
                "doc_id": c.doc_id,
                "source_filename": c.source_filename,
                "chunk_id": c.chunk_id,
                "position": idx,
            }
        )
    return embeddings, metadata
