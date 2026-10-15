"""Retrieval: baseline vector search (Phase 3) plus the Phase 9 upgrades.

Pipeline (each upgrade is a switch so we can measure it alone):
  [rewrite query] -> candidates (vector, or vector+BM25 fused) -> [rerank] -> top_k
"""
import re

from .utils import parse_json

RRF_K = 60  # standard constant from the original Reciprocal Rank Fusion paper


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def rrf_fuse(rankings: list[list[int]], limit: int) -> list[int]:
    """Reciprocal Rank Fusion: score(doc) = sum over lists of 1 / (60 + rank).

    Why RRF: BM25 scores and cosine scores live on different scales, so we cannot
    add them directly. Ranks are comparable, so we fuse ranks instead.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(scores, key=scores.get, reverse=True)[:limit]


class Retriever:
    def __init__(self, store, embedder, llm=None, top_k=5, pool=20,
                 hybrid=False, rerank=False, rewrite=False):
        if (rerank or rewrite) and llm is None:
            raise ValueError("rerank/rewrite need an llm")
        self.store, self.embedder, self.llm = store, embedder, llm
        self.top_k, self.pool = top_k, pool
        self.hybrid, self.rerank, self.rewrite = hybrid, rerank, rewrite
        self._bm25, self._bm25_size = None, -1

    # ---- upgrade 3: query rewriting -------------------------------------------------
    def rewrite_query(self, question: str) -> str:
        prompt = (
            "Rewrite this question as a short search query for finding passages in "
            "technical documents. Keep the key terms, expand abbreviations, and add "
            "likely synonyms. Output ONLY the query text.\n\nQuestion: " + question
        )
        out = self.llm.generate(prompt).strip().strip('"')
        return out or question  # fall back to the original if the model returns nothing

    # ---- upgrade 1: BM25 keyword search ---------------------------------------------
    def _bm25_ranking(self, query: str, limit: int) -> list[int]:
        from rank_bm25 import BM25Okapi  # imported here so baseline needs no extra dependency

        if self._bm25 is None or self._bm25_size != len(self.store.chunks):
            self._bm25 = BM25Okapi([tokenize(c["text"]) for c in self.store.chunks])
            self._bm25_size = len(self.store.chunks)
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [i for i in order[:limit] if scores[i] > 0]

    # ---- upgrade 2: LLM reranking ---------------------------------------------------
    def _rerank(self, question: str, chunks: list[dict]) -> tuple[list[dict], bool]:
        """Ask the LLM to order candidates by usefulness. Returns (chunks, ok)."""
        listing = "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(chunks, start=1))
        prompt = (
            f"Question: {question}\n\nPassages:\n{listing}\n\n"
            "Rank the passages by how useful they are for answering the question, most "
            'useful first. Respond with JSON only: {"ranking": [passage numbers]}'
        )
        data = parse_json(self.llm.generate(prompt, json_mode=True))
        if not data or not isinstance(data.get("ranking"), list):
            return chunks, False   # keep the original order, but REPORT the failure
        seen, ordered = set(), []
        for n in data["ranking"]:
            if isinstance(n, int) and 1 <= n <= len(chunks) and n not in seen:
                seen.add(n)
                ordered.append(chunks[n - 1])
        ordered += [c for i, c in enumerate(chunks, start=1) if i not in seen]
        return ordered, True

    # ---- main entry -----------------------------------------------------------------
    def retrieve(self, question: str):
        """Return (hits, info). hits = [{'chunk': dict, 'score': float|None}] best first."""
        info = {"query_used": question, "rerank_ok": None}
        query = question
        if self.rewrite:
            query = self.rewrite_query(question)
            info["query_used"] = query

        # candidate stage: take more than top_k so fusion/reranking has room to help
        pool = self.pool if (self.hybrid or self.rerank) else self.top_k
        vec_hits = self.store.search(self.embedder.embed_query(query), pool)
        scores = {c["id"]: s for c, s in vec_hits}

        if self.hybrid:
            vec_ids = [c["id"] for c, _ in vec_hits]
            kw_ids = self._bm25_ranking(query, pool)
            candidate_ids = rrf_fuse([vec_ids, kw_ids], pool)
        else:
            candidate_ids = [c["id"] for c, _ in vec_hits]

        candidates = [self.store.chunks[i] for i in candidate_ids]
        if self.rerank and candidates:
            candidates, ok = self._rerank(question, candidates)
            info["rerank_ok"] = ok

        hits = [{"chunk": c, "score": scores.get(c["id"])} for c in candidates[: self.top_k]]
        return hits, info
