from __future__ import annotations

from typing import List
import numpy as np


class EmbeddingClient:
    """Simple embedding client that returns deterministic random vectors."""

    def __init__(self, dim: int = 128):
        self.dim = dim

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            seed = abs(hash(text)) % (2**32)
            rng = np.random.default_rng(seed)
            vectors.append(rng.random(self.dim, dtype=np.float32))
        return np.vstack(vectors)
