from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class Chunk:
    doc_id: str
    source_filename: str
    chunk_id: int
    text: str


def chunk_documents(documents: Iterable, chunk_size: int = 200) -> List[Chunk]:
    """Split document text into smaller chunks by words."""
    chunks: List[Chunk] = []
    for doc in documents:
        words = doc.text.split()
        start = 0
        chunk_idx = 0
        while start < len(words):
            end = start + chunk_size
            chunk_text = " ".join(words[start:end])
            chunks.append(
                Chunk(
                    doc_id=doc.doc_id,
                    source_filename=doc.source_filename,
                    chunk_id=chunk_idx,
                    text=chunk_text,
                )
            )
            chunk_idx += 1
            start = end
    return chunks
