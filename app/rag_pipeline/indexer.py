from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np


def build_index(embeddings: np.ndarray) -> np.ndarray:
    """Return the embeddings as the index for simplicity."""
    return embeddings.astype("float32")


def save_index(index: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        np.save(f, index)


def save_metadata(metadata: List[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
