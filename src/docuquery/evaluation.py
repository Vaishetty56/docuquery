"""Phase 8: evaluation. Every number reported anywhere comes from summarize() on real run output.

Metrics
  retrieval hit@k   : is a gold (file, page) among the top-k retrieved chunks?  (no LLM involved)
  answer correctness: LLM judge compares answer to the gold answer. For questions the documents
                      cannot answer, the correct behaviour is to abstain (no judge needed).
  faithfulness      : LLM judge checks every claim in the answer is supported by the retrieved text.
The judges are models and can be wrong: hand-check a sample of their verdicts (see eval/README.md).
"""
import json
from pathlib import Path

from .utils import parse_json

KS = (1, 3, 5)

CORRECT_PROMPT = """You are grading an answer against a reference answer.
Question: {question}
Reference answer: {gold}
Candidate answer: {answer}

Verdict "correct" if the candidate states the key facts of the reference and does not contradict it
(extra correct detail is fine). Otherwise "incorrect".
Respond with JSON only: {{"verdict": "correct" or "incorrect", "reason": "one sentence"}}"""

FAITHFUL_PROMPT = """Passages:
{context}

Answer to check: {answer}

Is EVERY factual claim in the answer supported by the passages above? Ignore style.
Respond with JSON only: {{"verdict": "faithful" or "unfaithful", "reason": "one sentence"}}"""


def load_questions(path, verified_only=True) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return [r for r in rows if r.get("verified")] if verified_only else rows


def gold_pairs(q: dict) -> set[tuple[str, int]]:
    return {(q["gold_file"], p) for p in q.get("gold_pages", [])}


def hit_at_k(retrieved: list[dict], gold: set, k: int) -> bool:
    """True if ANY gold (file, page) appears in the top-k retrieved chunks."""
    return any((r["file"], r["page"]) in gold for r in retrieved[:k])


def full_hit_at_k(retrieved: list[dict], gold: set, k: int) -> bool:
    """True if ALL gold (file, page) pairs appear in the top-k (matters for multi-chunk questions)."""
    got = {(r["file"], r["page"]) for r in retrieved[:k]}
    return gold <= got


def _judge(llm, prompt: str, good: str):
    """Run a judge. Returns (True/False/None, reason). None = judge output unusable."""
    data = parse_json(llm.generate(prompt, json_mode=True))
    if not isinstance(data, dict) or "verdict" not in data:
        return None, "judge output unusable"
    return data["verdict"] == good, str(data.get("reason", ""))


def evaluate_question(dq, judge, q: dict, ks=KS) -> dict:
    res = dq.ask(q["question"])
    retrieved = [{"file": h["chunk"]["file"], "page": h["chunk"]["page"],
                  "chunk_id": h["chunk"]["id"], "score": h["score"]} for h in res["hits"]]
    row = {"id": q["id"], "type": q["type"], "question": q["question"],
           "gold_answer": q.get("gold_answer"), "gold_file": q.get("gold_file"),
           "gold_pages": q.get("gold_pages", []), "retrieved": retrieved,
           "answer": res["answer"], "abstained": not res["answerable"], "sources": res["sources"],
           "note": res["note"], "retrieval_info": res["retrieval_info"]}

    if q["type"] == "unanswerable":
        row["correct"], row["correct_reason"] = row["abstained"], "abstention expected"
    else:
        gold = gold_pairs(q)
        for k in ks:
            row[f"hit@{k}"] = hit_at_k(retrieved, gold, k)
            row[f"full_hit@{k}"] = full_hit_at_k(retrieved, gold, k)
        if row["abstained"]:
            row["correct"], row["correct_reason"] = False, "abstained on an answerable question"
        else:
            row["correct"], row["correct_reason"] = _judge(
                judge, CORRECT_PROMPT.format(question=q["question"], gold=q["gold_answer"],
                                             answer=res["answer"]), "correct")

    if not row["abstained"]:   # faithfulness only makes sense when an answer was given
        context = "\n\n".join(f"[{i}] {h['chunk']['text']}" for i, h in enumerate(res["hits"], 1))
        row["faithful"], row["faithful_reason"] = _judge(
            judge, FAITHFUL_PROMPT.format(context=context, answer=res["answer"]), "faithful")
    return row


def _frac(num: int, den: int) -> dict:
    return {"num": num, "den": den, "rate": (num / den) if den else None}


def summarize(rows: list[dict], ks=KS) -> dict:
    """Compute all metrics from raw rows. Each metric carries its numerator and denominator."""
    ok = [r for r in rows if not r.get("error")]
    ans = [r for r in ok if r["type"] != "unanswerable"]
    unans = [r for r in ok if r["type"] == "unanswerable"]
    multi = [r for r in ans if len(r["gold_pages"]) > 1]
    judged = [r for r in ans if r["correct"] is not None]
    answered = [r for r in ok if not r["abstained"] and r.get("faithful") is not None]
    out = {
        "n_questions": len(rows), "n_errors": len(rows) - len(ok),
        "n_answerable": len(ans), "n_unanswerable": len(unans),
        "retrieval_hit": {f"@{k}": _frac(sum(r[f"hit@{k}"] for r in ans), len(ans)) for k in ks},
        "retrieval_full_hit_multipage": {
            f"@{k}": _frac(sum(r[f"full_hit@{k}"] for r in multi), len(multi)) for k in ks},
        "correct_answerable": _frac(sum(r["correct"] for r in judged), len(judged)),
        "correct_abstention_on_unanswerable": _frac(sum(r["correct"] for r in unans), len(unans)),
        "faithfulness_of_given_answers": _frac(sum(r["faithful"] for r in answered), len(answered)),
        "judge_failures": sum(1 for r in ans if r["correct"] is None)
                          + sum(1 for r in ok if not r["abstained"] and r.get("faithful") is None),
        "rerank_failures": sum(1 for r in ok if r["retrieval_info"].get("rerank_ok") is False),
    }
    all_judged = judged + unans
    out["overall_correct"] = _frac(sum(r["correct"] for r in all_judged), len(all_judged))
    return out


def run_eval(dq, judge, questions, out_dir, run_name, run_config, resume=False) -> dict:
    """Run every question, appending raw results to disk as we go (so a crash loses nothing)."""
    out = Path(out_dir) / run_name
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "raw.jsonl"
    done = {}
    if resume and raw_path.exists():
        for line in raw_path.read_text().splitlines():
            r = json.loads(line)
            if not r.get("error"):
                done[r["id"]] = r
    rows = []
    with raw_path.open("w") as f:
        for q in questions:
            if q["id"] in done:
                row = done[q["id"]]
            else:
                try:
                    row = evaluate_question(dq, judge, q)
                except Exception as e:  # record and keep going; the error count is reported
                    row = {"id": q["id"], "type": q["type"], "question": q["question"],
                           "error": f"{type(e).__name__}: {e}"}
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {len(rows)}/{len(questions)} done: {q['id']}", flush=True)   # progress line
    summary = {"run_name": run_name, "config": run_config, "metrics": summarize(rows)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def classify_failure(row: dict, k: int = 5):
    """Deterministic failure label for a raw row, or None if the row is fine."""
    if row.get("error"):
        return "pipeline_error"
    if row["type"] == "unanswerable":
        return None if row["correct"] else "answered_unanswerable_question"
    if not row.get(f"hit@{k}"):
        return "retrieval_miss"
    if row["abstained"]:
        return "false_abstain_despite_retrieval_hit"
    if row["correct"] is False:
        return "wrong_answer_despite_retrieval_hit"
    if row.get("faithful") is False:
        return "unfaithful_answer"
    return None
