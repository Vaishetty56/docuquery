"""Phase 3a: text -> vectors using Gemini embeddings.

Free-tier note: Gemini limits embedding calls per minute (the limit we hit was 100 per
minute). We pace ourselves: never send more than `max_per_minute` texts in any 60-second
window, and wait (printing a message) when we would. We count every TEXT, not every
request, which is the safe assumption.
"""
import time
from collections import deque

import numpy as np
from google import genai
from google.genai import types

from .utils import with_retry


class GeminiEmbedder:
    def __init__(self, api_key: str, model: str, batch_size: int = 20, max_per_minute: int = 90,
                 clock=time.monotonic, sleep=time.sleep, log=print):
        if batch_size > max_per_minute:
            raise ValueError("batch_size must not exceed max_per_minute")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        self.max_per_minute = max_per_minute
        self.clock, self.sleep, self.log = clock, sleep, log
        self._sent = deque()   # (timestamp, number_of_texts) for the last 60 seconds
        # Per the docs, task_type is a gemini-embedding-001 feature. For other models
        # we do not set it (they may expect a text prefix instead; see Google's docs).
        self.use_task_type = model.endswith("embedding-001")

    def _wait_for_quota(self, n: int) -> None:
        """Block until sending n more texts keeps us within max_per_minute."""
        while True:
            now = self.clock()
            while self._sent and now - self._sent[0][0] >= 60:
                self._sent.popleft()
            if sum(c for _, c in self._sent) + n <= self.max_per_minute:
                return
            wait = 60 - (now - self._sent[0][0]) + 0.5
            self.log(f"  (free-tier rate limit: waiting {wait:.0f}s, this is normal)")
            self.sleep(max(wait, 0.5))

    def _embed(self, texts: list[str], task_type: str) -> np.ndarray:
        vectors = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            self._wait_for_quota(len(batch))
            cfg = types.EmbedContentConfig(task_type=task_type) if self.use_task_type else None
            self._sent.append((self.clock(), len(batch)))
            # If we still get a 429, wait 10s, 20s, 40s, 80s, 160s before giving up.
            resp = with_retry(
                lambda: self.client.models.embed_content(
                    model=self.model, contents=batch, config=cfg),
                tries=6, base_delay=10.0)
            vectors.extend(e.values for e in resp.embeddings)
            if len(texts) > self.batch_size:
                self.log(f"  embedded {min(i + self.batch_size, len(texts))}/{len(texts)} chunks")
        arr = np.array(vectors, dtype="float32")
        # L2-normalise so inner product == cosine similarity (what FAISS IndexFlatIP computes).
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(norms, 1e-12, None)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        # Documents and queries use different task types: the model is trained so a
        # short question lands near the passage that answers it.
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], "RETRIEVAL_QUERY")