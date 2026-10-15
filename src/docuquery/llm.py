"""Thin wrapper around Gemini text generation (used for answers, rerank, rewrite, judging).

Free-tier note: generation calls are rate limited too. We space calls at least
`min_interval` seconds apart and, if a 429 still happens, retry with long waits
(10s, 20s, 40s, 80s, 160s) before giving up.
"""
import time

from google import genai
from google.genai import types

from .utils import with_retry


class GeminiLLM:
    def __init__(self, api_key: str, model: str, min_interval: float = 4.0,
                 clock=time.monotonic, sleep=time.sleep):
                   # 120 s limit per request: a hung connection now fails and is retried instead of waiting forever
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=120_000))
        self.model = model
        self.min_interval = min_interval
        self.clock, self.sleep = clock, sleep
        self._last_call = None

    def _pace(self) -> None:
        """Wait so that calls are at least min_interval seconds apart."""
        if self._last_call is not None:
            wait = self.min_interval - (self.clock() - self._last_call)
            if wait > 0:
                self.sleep(wait)
        self._last_call = self.clock()

    def generate(self, prompt: str, system: str | None = None, json_mode: bool = False,
                 temperature: float = 0.0) -> str:
        """Return the model's text. temperature=0 keeps runs as repeatable as possible
        (not perfectly deterministic: LLM APIs can still vary slightly)."""
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            response_mime_type="application/json" if json_mode else None,
        )

        def call():
            self._pace()
            return self.client.models.generate_content(
                model=self.model, contents=prompt, config=config)

        resp = with_retry(call, tries=6, base_delay=10.0)
        return resp.text or ""   # .text can be empty if the response was blocked