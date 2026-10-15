"""Run the evaluation for one pipeline configuration.

  python eval/run_eval.py --name baseline
  python eval/run_eval.py --name hybrid --hybrid
  python eval/run_eval.py --name hybrid_rerank --hybrid --rerank
  python eval/run_eval.py --name hybrid_rerank_rewrite --hybrid --rerank --rewrite

By default only questions with "verified": true are used, so reported numbers rest on
labels a human checked. --include-unverified is for dry runs only; the summary records it.
"""
import argparse
import datetime
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docuquery.config import load_settings          # noqa: E402
from docuquery.evaluation import load_questions, run_eval  # noqa: E402
from docuquery.llm import GeminiLLM                 # noqa: E402
from docuquery.pipeline import DocuQuery            # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--questions", default=str(ROOT / "eval" / "questions.jsonl"))
    p.add_argument("--hybrid", action="store_true")
    p.add_argument("--rerank", action="store_true")
    p.add_argument("--rewrite", action="store_true")
    p.add_argument("--include-unverified", action="store_true")
    p.add_argument("--resume", action="store_true", help="reuse finished rows from a crashed run")
    a = p.parse_args()

    s = load_settings()
    dq = DocuQuery.from_settings(s, a.hybrid, a.rerank, a.rewrite)
    if not dq.store.chunks:
        sys.exit("Index is empty. Run: python -m docuquery.cli ingest data/pdfs")
    judge = GeminiLLM(s.api_key, s.judge_model)
    questions = load_questions(a.questions, verified_only=not a.include_unverified)
    if not questions:
        sys.exit("No questions to run (are any marked \"verified\": true?).")

    config = {"embed_model": s.embed_model, "gen_model": s.gen_model, "judge_model": s.judge_model,
              "chunk_size": s.chunk_size, "chunk_overlap": s.chunk_overlap, "top_k": s.top_k,
              "pool": s.pool, "hybrid": a.hybrid, "rerank": a.rerank, "rewrite": a.rewrite,
              "verified_labels_only": not a.include_unverified,
              "n_chunks_indexed": len(dq.store.chunks),
              "run_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    summary = run_eval(dq, judge, questions, ROOT / "eval" / "results", a.name, config, a.resume)
    m = summary["metrics"]
    print(f"Run '{a.name}' finished. Raw rows: eval/results/{a.name}/raw.jsonl")
    print(f"Questions: {m['n_questions']} ({m['n_errors']} errors). See summary.json for all metrics.")


if __name__ == "__main__":
    main()
