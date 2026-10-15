"""Glue: one object that ingests PDFs and answers questions. Used by the CLI, UI and eval."""
from pathlib import Path

from .config import ConfigError
from .embeddings import GeminiEmbedder
from .generate import answer_question
from .ingest import ingest_pdf
from .llm import GeminiLLM
from .retrieval import Retriever
from .store import VectorStore


class DocuQuery:
    def __init__(self, embedder, llm, store, chunk_size=1000, chunk_overlap=200,
                 index_dir=None, top_k=5, pool=20, hybrid=False, rerank=False, rewrite=False):
        self.embedder, self.llm, self.store = embedder, llm, store
        self.chunk_size, self.chunk_overlap = chunk_size, chunk_overlap
        self.index_dir = index_dir
        self.options = dict(top_k=top_k, pool=pool, hybrid=hybrid, rerank=rerank, rewrite=rewrite)
        self.retriever = Retriever(store, embedder, llm, **self.options)

    @classmethod
    def from_settings(cls, settings, hybrid=False, rerank=False, rewrite=False):
        embedder = GeminiEmbedder(settings.api_key, settings.embed_model)
        llm = GeminiLLM(settings.api_key, settings.gen_model)
        meta = {"embed_model": settings.embed_model, "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap}
        if VectorStore.exists(settings.index_dir):
            store = VectorStore.load(settings.index_dir)
            # Vectors from different embedding models are NOT comparable, so refuse to mix.
            if store.meta != meta:
                raise ConfigError(
                    f"Existing index was built with {store.meta} but settings are {meta}. "
                    "Delete data/index/ and re-ingest."
                )
        else:
            store = VectorStore(meta)
        return cls(embedder, llm, store, settings.chunk_size, settings.chunk_overlap,
                   settings.index_dir, settings.top_k, settings.pool, hybrid, rerank, rewrite)

    def add_pdf(self, pdf_path) -> dict:
        """Chunk, embed and index one PDF (skips files already indexed)."""
        name = Path(pdf_path).name
        if self.store.has_file(name):
            return {"file": name, "chunks": 0, "skipped": True, "empty_pages": []}
        chunks, empty = ingest_pdf(pdf_path, self.chunk_size, self.chunk_overlap)
        if chunks:
            self.store.add(chunks, self.embedder.embed_documents([c["text"] for c in chunks]))
        return {"file": name, "chunks": len(chunks), "skipped": False, "empty_pages": empty}

    def save(self) -> None:
        if self.index_dir:
            self.store.save(self.index_dir)

    def ask(self, question: str) -> dict:
        hits, info = self.retriever.retrieve(question)
        result = answer_question(question, hits, self.llm)
        result["hits"] = hits
        result["retrieval_info"] = info
        return result
