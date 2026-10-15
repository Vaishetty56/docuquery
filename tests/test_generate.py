"""Phase 4 tests: citations map to real pages; unsupported answers fall back."""
from conftest import FakeLLM
from docuquery.generate import NO_ANSWER, answer_question

HITS = [{"chunk": {"id": 0, "file": "a.pdf", "page": 7, "text": "Alpha fact."}, "score": 0.9},
        {"chunk": {"id": 1, "file": "b.pdf", "page": 2, "text": "Beta fact."}, "score": 0.8}]


def ask(resp):
    return answer_question("q?", HITS, FakeLLM(lambda p, s, j: resp))


def test_valid_answer_maps_citations_to_file_and_page():
    r = ask({"answerable": True, "answer": "Alpha.", "citations": [1]})
    assert r["answerable"] and r["sources"] == [{"file": "a.pdf", "page": 7, "chunk_id": 0}]


def test_duplicate_page_citations_are_merged():
    r = ask({"answerable": True, "answer": "x", "citations": [1, 1]})
    assert len(r["sources"]) == 1


def test_not_answerable_gives_fallback():
    r = ask({"answerable": False, "answer": "", "citations": []})
    assert r["answer"] == NO_ANSWER and not r["answerable"] and r["sources"] == []


def test_answer_without_valid_citation_is_rejected():
    for cites in ([], [99], ["x"]):
        r = ask({"answerable": True, "answer": "Made up.", "citations": cites})
        assert r["answer"] == NO_ANSWER and "citation" in r["note"]


def test_invalid_json_falls_back():
    r = ask("sorry, I cannot do JSON")
    assert r["answer"] == NO_ANSWER and "JSON" in r["note"]


def test_string_false_is_not_treated_as_true():
    r = ask({"answerable": "false", "answer": "x", "citations": [1]})
    assert not r["answerable"]


def test_no_hits_does_not_call_llm():
    llm = FakeLLM(lambda p, s, j: {})
    assert answer_question("q", [], llm)["answer"] == NO_ANSWER and llm.calls == []


def test_code_fenced_json_is_accepted():
    r = ask('```json\n{"answerable": true, "answer": "ok", "citations": [2]}\n```')
    assert r["sources"][0]["file"] == "b.pdf"
