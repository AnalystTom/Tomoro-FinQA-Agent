from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Document:
    doc_id: str
    source_filename: str
    text: str


def parse_train_json(path: str | Path) -> List[Document]:
    """Load train.json and return list of Documents."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    documents: List[Document] = []
    for item in data:
        doc_id = item.get("doc_id") or item.get("id")
        source = item.get("source_filename") or item.get("file")
        pre_text = item.get("pre_text", "")
        post_text = item.get("post_text", "")
        table = item.get("table", [])
        table_str = "\n".join([" | ".join(row) for row in table])
        combined = "\n".join(filter(None, [pre_text, table_str, post_text]))
        documents.append(Document(doc_id=doc_id, source_filename=source, text=combined))
    return documents
