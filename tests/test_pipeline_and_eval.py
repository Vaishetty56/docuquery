"""End-to-end (offline) and evaluation-math tests."""
import json

from conftest import FakeEmbedder, FakeLLM, make_pdf
from docuquery.evaluation import (classify_failure, evaluate_question, full_hit_at_k, hit_at_k,
                                  load_questions, run_eval, summarize)
from docuquery.pipeline import DocuQuery
from docuquery.store import VectorStore


def build(tmp_path, llm):
    pdf = make_pdf(tmp_path / "doc.pdf", [
        "Governance means policies and accountability are in place for AI risk.",
        "Measurement uses quantitative and qualitative tools to assess AI risk."])
    dq = DocuQuery(FakeEmbedder(), llm, VectorStore({"embed_model": "fake"}),
                   index_dir=tmp_path / "idx", top_k=2)
    dq.add_pdf(pdf)
    return dq


def test_end_to_end_citation_points_to_right_page(tmp_path):
    llm = FakeLLM(lambda p, s, j: {"answerable": True, "answer": "Measurement tools.", "citations": [1]})
    dq = build(tmp_path, llm)
    res = dq.ask("What does measurement use to assess risk?")
    assert res["sources"][0]["file"] == "doc.pdf" and res["sources"][0]["page"] == 2


def test_adding_same_pdf_twice_is_skipped(tmp_path):
    dq = build(tmp_path, FakeLLM(lambda p, s, j: {}))
    assert dq.add_pdf(tmp_path / "doc.pdf")["skipped"] is True


def test_index_persists_to_disk(tmp_path):
    dq = build(tmp_path, FakeLLM(lambda p, s, j: {}))
    dq.save()
    assert len(VectorStore.load(tmp_path / "idx").chunks) == len(dq.store.chunks)


def test_hit_metrics():
    retrieved = [{"file": "a", "page": 1}, {"file": "a", "page": 2}, {"file": "b", "page": 9}]
    gold = {("a", 2), ("b", 9)}
    assert not hit_at_k(retrieved, gold, 1) and hit_at_k(retrieved, gold, 2)
    assert not full_hit_at_k(retrieved, gold, 2) and full_hit_at_k(retrieved, gold, 3)


def test_unanswerable_correct_only_if_abstained(tmp_path):
    q = {"id": "u1", "type": "unanswerable", "question": "What is the CEO's shoe size?"}
    abstain = FakeLLM(lambda p, s, j: {"answerable": False, "answer": "", "citations": []})
    row = evaluate_question(build(tmp_path, abstain), abstain, q)
    assert row["correct"] is True and "faithful" not in row


def test_answerable_flow_uses_judges(tmp_path):
    def responder(prompt, system, json_mode):
        if "grading an answer" in prompt:
            return {"verdict": "correct", "reason": "matches"}
        if "Is EVERY factual claim" in prompt:
            return {"verdict": "faithful", "reason": "ok"}
        return {"answerable": True, "answer": "Measurement tools.", "citations": [1]}
    llm = FakeLLM(responder)
    q = {"id": "q1", "type": "single", "question": "What does measurement use to assess risk?",
         "gold_answer": "Quantitative and qualitative tools", "gold_file": "doc.pdf", "gold_pages": [2]}
    row = evaluate_question(build(tmp_path, llm), llm, q)
    assert row["hit@1"] and row["correct"] is True and row["faithful"] is True


def _row(**kw):
    base = {"id": "x", "type": "single", "gold_pages": [1], "abstained": False, "correct": True,
            "faithful": True, "retrieval_info": {}, "hit@1": True, "hit@3": True, "hit@5": True,
            "full_hit@1": True, "full_hit@3": True, "full_hit@5": True}
    return {**base, **kw}


def test_summarize_counts_and_denominators():
    rows = [_row(), _row(**{"hit@1": False}, correct=False),
            _row(type="unanswerable", gold_pages=[], correct=True, abstained=True),
            {"id": "e", "type": "single", "error": "boom"}]
    m = summarize(rows)
    assert m["n_errors"] == 1 and m["n_answerable"] == 2 and m["n_unanswerable"] == 1
    assert m["retrieval_hit"]["@1"] == {"num": 1, "den": 2, "rate": 0.5}
    assert m["correct_abstention_on_unanswerable"]["den"] == 1
    assert m["overall_correct"] == {"num": 2, "den": 3, "rate": 2 / 3}


def test_failure_classification():
    assert classify_failure(_row(**{"hit@5": False})) == "retrieval_miss"
    assert classify_failure(_row(abstained=True, correct=False)) == "false_abstain_despite_retrieval_hit"
    assert classify_failure(_row(correct=False)) == "wrong_answer_despite_retrieval_hit"
    assert classify_failure(_row(faithful=False)) == "unfaithful_answer"
    assert classify_failure(_row(type="unanswerable", correct=False)) == "answered_unanswerable_question"
    assert classify_failure(_row()) is None


def test_run_eval_writes_raw_and_summary(tmp_path):
    def responder(prompt, system, json_mode):
        if "grading an answer" in prompt:
            return {"verdict": "correct", "reason": ""}
        if "Is EVERY factual claim" in prompt:
            return {"verdict": "faithful", "reason": ""}
        return {"answerable": True, "answer": "A.", "citations": [1]}
    llm = FakeLLM(responder)
    qs = [{"id": "q1", "type": "single", "question": "measurement risk tools", "gold_answer": "A",
           "gold_file": "doc.pdf", "gold_pages": [2], "verified": True}]
    s = run_eval(build(tmp_path, llm), llm, qs, tmp_path / "res", "t", {"x": 1})
    assert (tmp_path / "res" / "t" / "raw.jsonl").exists()
    assert json.loads((tmp_path / "res" / "t" / "summary.json").read_text())["metrics"]["n_questions"] == 1
    assert s["metrics"]["correct_answerable"]["num"] == 1


def test_load_questions_ignores_unverified(tmp_path):
    f = tmp_path / "q.jsonl"
    f.write_text(json.dumps({"id": "a", "verified": True}) + "\n" + json.dumps({"id": "b", "verified": False}) + "\n")
    assert [q["id"] for q in load_questions(f)] == ["a"]
    assert len(load_questions(f, verified_only=False)) == 2
