"""Command line: ingest PDFs and ask questions without the UI.

  python -m docuquery.cli ingest data/pdfs
  python -m docuquery.cli ask "What is the GOVERN function?" [--hybrid --rerank --rewrite]
"""
import argparse
from pathlib import Path

from .config import load_settings
from .pipeline import DocuQuery


def main():
    p = argparse.ArgumentParser(prog="docuquery")
    sub = p.add_subparsers(dest="cmd", required=True)
    ing = sub.add_parser("ingest", help="index every PDF in a folder (or one PDF)")
    ing.add_argument("path")
    ask = sub.add_parser("ask", help="ask a question")
    ask.add_argument("question")
    for flag in ("hybrid", "rerank", "rewrite"):
        ask.add_argument(f"--{flag}", action="store_true")
    args = p.parse_args()

    settings = load_settings()
    if args.cmd == "ingest":
        dq = DocuQuery.from_settings(settings)
        target = Path(args.path)
        pdfs = sorted(target.glob("*.pdf")) if target.is_dir() else [target]
        for pdf in pdfs:
            r = dq.add_pdf(pdf)
            status = "skipped (already indexed)" if r["skipped"] else f"{r['chunks']} chunks"
            warn = f"  WARNING: {len(r['empty_pages'])} pages had no text" if r["empty_pages"] else ""
            print(f"{r['file']}: {status}{warn}")
        dq.save()
        print(f"Index now holds {len(dq.store.chunks)} chunks.")
    else:
        dq = DocuQuery.from_settings(settings, args.hybrid, args.rerank, args.rewrite)
        res = dq.ask(args.question)
        print(res["answer"])
        for s in res["sources"]:
            print(f"  - {s['file']}, page {s['page']}")
        if res["note"]:
            print(f"[note: {res['note']}]")


if __name__ == "__main__":
    main()
