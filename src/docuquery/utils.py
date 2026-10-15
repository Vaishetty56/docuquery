"""Small helpers: retry with backoff, and forgiving JSON parsing."""
import json
import time


def with_retry(fn, tries: int = 5, base_delay: float = 2.0):
    """Call fn(); on any exception wait 2s, 4s, 8s... and retry.

    Why: Gemini rate limits (HTTP 429) and brief network errors are normal,
    especially on free tiers. After the last try we re-raise so failures are visible.
    """
    for attempt in range(tries):
        try:
            return fn()
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))


def parse_json(text: str):
    """Parse a JSON object from model output. Returns None if it cannot."""
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):  # models sometimes wrap JSON in code fences
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None
