# Phase notes: concepts, interview questions, how to verify

(Phases 1-10. Read each phase's file alongside it. Where I say "run", the command needs your `.env`.)

## Phase 1: Plan and scaffold  (`PLAN.md`, `requirements.txt`, `.env.example`, `.gitignore`, `config.py`)
**Concepts.** Secrets live in environment variables, never in code. Model names are configuration, so an
eval result can record exactly which models produced it. Pinned versions make the repo reproducible.
**Interviewer may ask**
1. *Why keep model names in .env?* Results depend on the exact model; config lets me swap and record them without code edits.
2. *How do you protect the API key?* Git-ignored `.env`, placeholder-only `.env.example`; rotate the key if it ever reaches git history, since deleting a commit does not un-leak it.
3. *Why pin dependencies?* Library updates change behaviour; pinning makes my numbers reproducible.
**Verify.** `pytest -q` (all tests pass offline); `git status` must not list `.env`.

## Phase 2: Ingestion and chunking  (`ingest.py`)
**Concepts.** A model can only read a few passages, so we split documents into chunks and retrieve the
relevant ones. I use **1000 characters (~150-200 words)** per chunk: big enough to hold one self-contained
idea, small enough that a match is precise and 5 chunks fit comfortably in a prompt. **200 characters of
overlap (20%)** so a sentence cut at a boundary appears whole in at least one chunk. Chunks end at sentence
or word boundaries and **never cross a page**, so every chunk has exactly one page for citation.
These values are reasonable defaults, **not tuned**. `CHUNK_SIZE`/`CHUNK_OVERLAP` in `.env` let you test
alternatives with the eval (re-ingest after changing). Scanned pages have no text; we warn instead of failing silently.
**Interviewer may ask**
1. *Why 1000/200?* Reasoned default, not optimised; too small loses context, too large dilutes the match and wastes prompt space. I can test other values with my eval.
2. *Downside of never crossing pages?* Answers spanning a page break are split; that is what my multi-chunk questions probe.
3. *What about tables or scans?* pypdf extracts tables as flat text and cannot read scanned images; I warn on empty pages. OCR would be the next step.
**Verify.** `pip install -e .` then `python -m docuquery.cli ingest data/pdfs` prints chunks per file.

## Phase 3: Embeddings and vector search  (`embeddings.py`, `store.py`)
**Concepts.** An embedding turns text into a vector so that similar meaning means nearby vectors. We
L2-normalise so inner product equals cosine similarity, and FAISS `IndexFlatIP` does exact search.
Documents and queries use different task types. FAISS returns row numbers; `chunks.json` maps them to text/file/page.
The index records which embedding model built it and refuses to mix models (vectors from different models are not comparable).
**Interviewer may ask**
1. *Why normalise vectors?* It makes dot product equal cosine similarity, so scores are comparable and FAISS can use inner product.
2. *Why exact search, not an approximate index?* At thousands of chunks brute force is fast; approximate indexes (IVF/HNSW) trade accuracy for speed only at much larger scale.
3. *What if you change the embedding model?* All chunks must be re-embedded; vectors from different models live in different spaces.
**Verify.** After ingest, `python -m docuquery.cli ask "your question"` should print sources even before generation is trusted.

## Phase 4: Answer generation  (`generate.py`)
**Concepts.** The prompt gives numbered passages and the rule "answer only from these". The model returns JSON
`{answerable, answer, citations:[numbers]}`. **We** convert numbers to (file, page), so the model cannot invent a page.
If the model says not answerable, returns bad JSON, or gives an answer with no valid citation, we show the "I don't know" fallback.
Limit: this checks that a citation *exists*, not that the passage truly supports the claim; the faithfulness eval measures that.
**Interviewer may ask**
1. *How do you prevent hallucinated citations?* The model cites passage numbers only; code maps them to real pages and rejects invalid ones.
2. *Does a citation prove the answer is right?* No, only that the model pointed at a passage; I measure faithfulness separately.
3. *How does the fallback work, and what are its failure modes?* Model-declared unanswerable plus code-enforced citation rule; failure modes are false abstentions and confident wrong answers, both counted in the eval.
**Verify.** Ask something your PDFs do not cover; expect the fallback line. Ask something they do; expect an answer with sources.

## Phase 5: Streamlit UI  (`app.py`)
**Concepts.** Streamlit reruns the whole script on every interaction, so the loaded index is cached with `st.cache_resource`. The sidebar toggles the Phase 9 upgrades. Uploaded file names are stripped to a bare name.
**Interviewer may ask**
1. *Why cache_resource?* Otherwise the index reloads on every click.
2. *Security issues with uploads?* Path traversal via file name (I strip to the base name), and file size / malicious PDFs (not handled; known limitation).
3. *How would you scale it to many users?* Per-user indexes, a real vector DB, async ingestion, auth; this is a single-user local app.
**Verify.** `streamlit run app.py`, upload a PDF, ask a question, open "Retrieved passages".

## Phase 6: Dataset  (`DATASET.md`)
**Concepts.** A dataset you may legally use and can describe. Sources and licence basis are recorded; confirm each PDF yourself.
**Interviewer may ask**
1. *Why this dataset?* One focused topic, public documents, clear provenance.
2. *Licence?* US-government works (not CC); public domain in the US, foreign rights reserved; I link rather than redistribute.
3. *Bias?* One publisher and style; findings may not generalise.

## Phase 7: Evaluation set  (`eval/draft_questions.py`, `eval/README.md`)
**Concepts.** Gold questions with gold answers and gold pages. I draft candidates with an LLM, then **a human verifies each**; only rows marked `verified: true` are scored. Question types: single-chunk, multi-chunk, unanswerable. Drafted questions are biased toward easy retrieval, so I also write some myself.
**Interviewer may ask**
1. *How were labels made?* LLM-drafted, then hand-verified against the PDFs; unverified rows are excluded from scoring.
2. *Why include unanswerable questions?* To measure whether the system refuses instead of hallucinating.
3. *Risk of the test set?* Small size (wide error bars), drafting bias, and label mistakes.

## Phase 8: Baseline evaluation  (`evaluation.py`, `eval/run_eval.py`, `eval/report.py failures`)
**Concepts.** *hit@k*: gold page appears in the top-k retrieved. *Correctness*: LLM judge vs gold answer (abstaining on an answerable question counts wrong; abstaining on an unanswerable one counts right). *Faithfulness*: LLM judge checks every claim is supported by retrieved text. Raw per-question results and a summary are saved. Failures are labelled deterministically: retrieval miss, false abstain, wrong answer despite a hit, unfaithful, answered an unanswerable question.
**Interviewer may ask**
1. *Why hit@k and not just answer accuracy?* It separates retrieval errors from generation errors, so I know what to fix.
2. *Can you trust an LLM judge?* Not blindly: I hand-check samples of verdicts and ideally use a different judge model.
3. *Your biggest eval weakness?* Test-set size; one question shifts the score by 100/N points.

## Phase 9: Upgrades  (`retrieval.py`)
**Concepts.** *Hybrid*: BM25 (keyword) + vector search, merged with Reciprocal Rank Fusion (rank-based, so no score scaling issue). *Rerank*: fetch 20 candidates, ask the LLM to order them, keep 5. *Query rewrite*: LLM turns the question into a keyword-rich search query. Each is a switch; run the same eval after each, **cumulatively**, and compare. An upgrade that does not help is a valid, reportable result.
**Interviewer may ask**
1. *Why does hybrid help?* Embeddings can miss exact terms (acronyms, IDs); BM25 catches those, vectors catch paraphrases.
2. *Why RRF?* BM25 and cosine scores are on different scales; ranks are comparable.
3. *Cost of reranking and rewriting?* Extra LLM calls: latency and money per query.
**Verify.** `python eval/run_eval.py --name hybrid --hybrid`, etc., then `python eval/report.py table baseline hybrid hybrid_rerank hybrid_rerank_rewrite`.

## Phase 10: README  (`README.md`)
The results block in the README is filled by `report.py table ... --readme` from saved summaries, never typed by hand.
