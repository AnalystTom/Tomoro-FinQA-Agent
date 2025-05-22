from pathlib import Path
import importlib

from scripts import build_vector_store


def test_build_vector_store(tmp_path, monkeypatch):
    # Redirect output paths to a temporary directory
    idx = tmp_path / "faiss_index.idx"
    meta = tmp_path / "metadata_mapping.json"

    monkeypatch.setattr(build_vector_store, "INDEX_PATH", idx)
    monkeypatch.setattr(build_vector_store, "METADATA_PATH", meta)

    build_vector_store.main()

    assert idx.exists()
    assert meta.exists()
