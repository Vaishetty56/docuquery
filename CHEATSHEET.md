# Interview cheat sheet: 20 questions

Fill every `[ ]` from YOUR `eval/results/*/summary.json`. Do not say a number you did not measure.

**1. What does DocuQuery do?** Upload PDFs, ask questions, get answers grounded in the text with file-and-page citations; if the documents don't support an answer it says "I don't know".

**2. Walk me through the pipeline.** Extract text per page, chunk (1000 chars, 200 overlap, never across pages), embed with Gemini, store in FAISS with a JSON sidecar for text/file/page. At query time: embed the question, retrieve top-k, build a prompt of numbered passages, Gemini returns JSON with answer and passage numbers, code maps numbers to pages.

**3. Why RAG instead of fine-tuning or pasting whole PDFs?** Documents change (re-index instead of retrain), answers can be cited and checked, and long inputs are costly and models can lose details in them.

**4. Why FAISS, and why a JSON sidecar?** FAISS stores only vectors and returns row numbers; the sidecar maps row i to chunk text, file and page. I picked it to understand the index; Chroma would handle metadata for me.

**5. How did you choose chunk size and overlap?** Reasoned defaults (1000/200), not optimised. I can test alternatives with the eval. [If you ran one, state the measured result here; otherwise say you did not.]

**6. How do you stop hallucinated citations?** The model cites passage numbers only; my code maps them to real (file, page) and rejects answers with no valid citation. That proves the passage exists, not that it supports the claim, which is why I also measure faithfulness.

**7. How does the "I don't know" fallback work, and what are its failure modes?** The model must set `answerable` false when passages don't support an answer; code also rejects uncited or unparsable output. Failure modes: false abstention on answerable questions and confident answers to unanswerable ones. Measured: correct abstention [ ]/[ ] on unanswerable, false abstentions [ ] (see failures.md).

**8. How big is your test set and how were labels made?** [N] questions: [a] answerable (single/multi-chunk) and [u] unanswerable. Drafted by an LLM from the indexed text, then I verified each against the PDFs; only verified rows are scored. [N_self] questions I wrote myself.

**9. What does "accuracy [ ]%" actually mean?** Be precise: [overall_correct num/den], where answerable answers were judged by an LLM against my gold answer and unanswerable ones count correct only if the system abstained. Say which number you quote and its denominator.

**10. Why trust an LLM judge?** I don't blindly. I hand-read [ ] correctness and [ ] faithfulness verdicts and agreed with [ ]; judge model was [ ] (different from/same as generator). Known bias: self-preference when they match.

**11. What are the error bars?** With N=[ ], one question moves the score by [100/N] points; differences between configurations smaller than a few questions are not evidence. I report counts (x/y), not just percentages.

**12. What did retrieval hit@k look like and why does it matter?** hit@1/3/5 = [ ]/[ ]/[ ]. It isolates retrieval from generation: a miss means the answer was never available to the model.

**13. What were your main failure modes?** [From failures.md: counts per category, one concrete example each, and what you think caused it after reading the retrieved text.]

**14. What improved results?** [Per-upgrade deltas from the results table. If an upgrade did not help or hurt, say so and offer the reason you checked, e.g. rerank_failures count.]

**15. Why hybrid search, and why RRF?** Embeddings can miss exact terms like acronyms; BM25 catches them. Their scores aren't on the same scale, so RRF fuses ranks.

**16. What does reranking cost?** One extra LLM call over 20 passages per question: latency and money. [State measured effect if you timed it; otherwise don't claim.]

**17. What would break this in production?** Scanned PDFs (no OCR), tables/figures extracted as flat text, no auth or per-user isolation, no size limits, rate limits on the Gemini free tier, index rebuild required on embedding model change.

**18. How would you improve it next?** OCR, layout-aware parsing for tables, a larger independent test set, a cross-encoder reranker compared with the LLM reranker, tuning chunk size/k on a held-out split.

**19. How is the API key handled and what did you test?** Env var via git-ignored `.env`; offline unit tests use a fake embedder and fake LLM so tests need no key. The eval itself needs the real API.

**20. What would you do differently?** Write more of the questions myself from the start (drafted ones favour easy retrieval), verify labels before building features, and log per-query latency and cost from day one.
