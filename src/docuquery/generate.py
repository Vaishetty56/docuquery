"""Phase 4: grounded answer generation with citations and an 'I don't know' fallback.

Citation design: the model cites PASSAGE NUMBERS ([1], [2]...). WE map those numbers
back to (file, page). The model never writes file names or page numbers itself,
so it cannot invent a page.
"""
from .utils import parse_json

NO_ANSWER = "I don't know based on the uploaded documents."

SYSTEM = """You answer questions using ONLY the numbered passages provided.
Rules:
- If the passages do not contain enough information to answer, set "answerable" to false.
- Never use outside knowledge and never guess.
- List the numbers of the passages that support your answer in "citations".
Respond with JSON only: {"answerable": true or false, "answer": "...", "citations": [1, 2]}"""


def build_prompt(question: str, hits: list[dict]) -> str:
    passages = "\n\n".join(f"[{i}]\n{h['chunk']['text']}" for i, h in enumerate(hits, start=1))
    return f"Passages:\n{passages}\n\nQuestion: {question}"


def _fallback(raw: str, reason: str) -> dict:
    return {"answer": NO_ANSWER, "answerable": False, "sources": [], "raw": raw, "note": reason}


def answer_question(question: str, hits: list[dict], llm) -> dict:
    """Return {answer, answerable, sources[{file,page,chunk_id}], raw, note}."""
    if not hits:
        return _fallback("", "no passages retrieved")

    raw = llm.generate(build_prompt(question, hits), system=SYSTEM, json_mode=True)
    data = parse_json(raw)
    if not isinstance(data, dict):
        return _fallback(raw, "model output was not valid JSON")

    answer = str(data.get("answer", "")).strip()
    if data.get("answerable") is not True or not answer:
        return _fallback(raw, "model said the passages do not answer the question")

    # keep only citations that point at a real passage number
    cited = [n for n in data.get("citations", []) if isinstance(n, int) and 1 <= n <= len(hits)]
    if not cited:
        # An uncited answer cannot be checked, so we refuse to show it.
        return _fallback(raw, "answer had no valid citation, rejected")

    sources, seen = [], set()
    for n in cited:
        chunk = hits[n - 1]["chunk"]
        key = (chunk["file"], chunk["page"])
        if key not in seen:
            seen.add(key)
            sources.append({"file": chunk["file"], "page": chunk["page"], "chunk_id": chunk["id"]})
    return {"answer": answer, "answerable": True, "sources": sources, "raw": raw, "note": ""}
