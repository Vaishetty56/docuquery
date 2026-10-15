"""Phase 3b: FAISS index + a JSON 'sidecar' holding the chunk text/file/page.

FAISS only knows vectors and returns ROW NUMBERS. Row i of the index corresponds
to chunks[i]. That mapping is what lets us turn a search hit into a citation.
"""
import json
from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    def __init__(self, meta: dict | None = None):
        self.index = None             # created on first add (we learn the dimension then)
        self.chunks: list[dict] = []  # chunks[i] <-> vector row i
        self.meta = meta or {}        # embed model + chunk settings used to build this index

    def has_file(self, file_name: str) -> bool:
        return any(c["file"] == file_name for c in self.chunks)

    def add(self, chunks: list[dict], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be the same length")
        if not len(chunks):
            return
        if self.index is None:
            # IndexFlatIP = exact (brute-force) inner-product search. Fine for
            # thousands of chunks; approximate indexes only matter at millions.
            self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        for c in chunks:
            self.chunks.append({**c, "id": len(self.chunks)})

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[dict, float]]:
        if self.index is None or not self.chunks:
            return []
        scores, ids = self.index.search(query_vec, min(k, len(self.chunks)))
        return [(self.chunks[i], float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]

    def save(self, directory) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, str(d / "index.faiss"))
        # utf-8 explicitly: Windows defaults to cp1252, which cannot store characters like "fi" ligatures
        (d / "chunks.json").write_text(json.dumps(self.chunks, ensure_ascii=False), encoding="utf-8")
        (d / "meta.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory) -> "VectorStore":
        d = Path(directory)
        store = cls(json.loads((d / "meta.json").read_text(encoding="utf-8")))
        store.chunks = json.loads((d / "chunks.json").read_text(encoding="utf-8"))
        if (d / "index.faiss").exists():
            store.index = faiss.read_index(str(d / "index.faiss"))
        return store

    @staticmethod
    def exists(directory) -> bool:
        return (Path(directory) / "chunks.json").exists()