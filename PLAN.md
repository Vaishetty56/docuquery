# DocuQuery: project plan

**Goal:** upload PDFs, ask questions, get answers grounded in the documents,
with citations (file name + page) and an "I don't know" fallback.

## Design decisions (confirmed by you)
| Decision | Choice | Why |
|---|---|---|
| Vector store | FAISS + JSON sidecar | Learn what an index really does; easy to add BM25 later |
| Orchestration | Plain Python | Every file explainable; upgrades measurable in isolation |
| Models | Stable Gemini models, names in `.env` | Reproducible evals; swap without code changes |

## Decisions I made without asking (veto any of these)
| Decision | Choice | Tradeoff |
|---|---|---|
| Chunking | 1000 chars, 200 overlap, never crossing a page | Exact page citations vs. ideas split across pages |
| Reranker | Gemini as LLM-reranker (one call per question) | No extra model download; costs an API call and adds latency. A cross-encoder is the common alternative |
| Judges | Gemini judges correctness + faithfulness | Cheap and scalable, but a model can misjudge, so you hand-check samples |
| Eval labels | LLM drafts, human verifies; only verified rows are scored | Defensible numbers vs. smaller test set |
| Dataset | NIST trustworthy-AI publications | Not CC-licensed; public domain in the US (see DATASET.md) |

## Pipeline
```
PDF -> pypdf (text per page) -> chunker (keeps file + page)
    -> Gemini embeddings -> FAISS index + chunks.json (text, file, page)

question -> [rewrite] -> vector search (+ BM25 fused by RRF) -> [LLM rerank] -> top-k
         -> prompt (numbered passages + rules) -> Gemini -> JSON {answerable, answer, citations}
         -> we map citation numbers to (file, page); no valid citation => "I don't know"
```

## Repo layout
```
src/docuquery/  config.py utils.py ingest.py llm.py embeddings.py store.py
                retrieval.py generate.py pipeline.py cli.py evaluation.py
app.py          Streamlit UI
eval/           run_eval.py draft_questions.py report.py make_bullets.py README.md
tests/          offline tests (fake embedder + fake LLM, no API key needed)
```
