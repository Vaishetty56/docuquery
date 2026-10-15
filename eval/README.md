# Evaluation guide

**Schema** (one JSON object per line in `questions.jsonl`):
```json
{"id": "q001", "type": "single | multi_chunk | unanswerable",
 "question": "...", "gold_answer": "...", "gold_file": "NIST.AI.100-1.pdf",
 "gold_pages": [14], "verified": true, "notes": ""}
```
For `unanswerable`, leave `gold_answer`/`gold_file` null and `gold_pages` empty.
`gold_pages` are PHYSICAL pages: the number your PDF viewer shows in its page box,
not the page number printed on the page (they often differ).

**Workflow**
1. `python eval/draft_questions.py` -> `questions.draft.jsonl` (all `verified: false`)
2. Open the PDFs and check every row you intend to use. Fix wrong answers/pages. Delete bad rows.
3. Add 10+ questions you write yourself (drafted ones are biased towards easy retrieval).
4. Copy checked rows into `questions.jsonl` with `"verified": true`.
5. Run the four configurations (see `run_eval.py`), then `report.py failures <run>` and `report.py table ...`.

**Which labels you MUST hand-check**
- Everything in `questions.jsonl`. Your headline numbers are only as good as these labels, and the
  eval ignores rows not marked verified. You said you'd check at least 15; if you verify only 15,
  your test set is 15 questions and your claims must say so. More verified = more defensible.
- Highest priority if time is short: (1) every `unanswerable` row (search all PDFs to confirm the
  answer really is absent, since a wrong label makes a correct answer look like a failure);
  (2) every `multi_chunk` row (are both pages truly needed?); (3) any row whose gold page is near a
  page boundary (the answer text may sit on the neighbouring page); (4) a random sample of the rest.

**Judges are models.** After each run, hand-read at least 10 correctness verdicts and 10 faithfulness
verdicts in `raw.jsonl` (`correct_reason`, `faithful_reason`). If you disagree with several, say so in
the README. Prefer a `JUDGE_MODEL` different from `GEN_MODEL`.

**Honest limits**: small test set means wide uncertainty (one question moves a 40-question score by 2.5
points); runs are not perfectly repeatable; chunk size / k / prompts were not tuned on a held-out set.
