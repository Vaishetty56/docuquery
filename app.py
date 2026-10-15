"""Phase 5: Streamlit UI.   Run:  streamlit run app.py"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from docuquery.config import ConfigError, load_settings  # noqa: E402
from docuquery.pipeline import DocuQuery                  # noqa: E402

st.set_page_config(page_title="DocuQuery", page_icon="📄")
st.title("📄 DocuQuery")
st.caption("Ask questions about your PDFs. Answers are grounded in the documents and cite file + page.")

try:
    settings = load_settings()
except ConfigError as e:
    st.error(str(e))
    st.stop()

# Retrieval upgrades can be toggled to see their effect (they are what the eval compares).
with st.sidebar:
    st.header("Documents")
    hybrid = st.checkbox("Hybrid search (BM25 + vectors)")
    rerank = st.checkbox("LLM reranking")
    rewrite = st.checkbox("Query rewriting")


@st.cache_resource
def get_app(hybrid: bool, rerank: bool, rewrite: bool) -> DocuQuery:
    # Cached so the index is loaded once, not on every click (Streamlit reruns the script each time).
    return DocuQuery.from_settings(settings, hybrid, rerank, rewrite)


try:
    dq = get_app(hybrid, rerank, rewrite)
except ConfigError as e:
    st.error(str(e))
    st.stop()

with st.sidebar:
    uploads = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if uploads and st.button("Add to index"):
        settings.pdf_dir.mkdir(parents=True, exist_ok=True)
        for up in uploads:
            name = Path(up.name).name            # strip any path parts from the uploaded name
            path = settings.pdf_dir / name
            path.write_bytes(up.getvalue())
            with st.spinner(f"Indexing {name}..."):
                r = dq.add_pdf(path)
            if r["skipped"]:
                st.info(f"{name}: already indexed")
            else:
                st.success(f"{name}: {r['chunks']} chunks")
            if r["empty_pages"]:
                st.warning(f"{name}: {len(r['empty_pages'])} pages had no text (scanned?). They can't be searched.")
        dq.save()
    files = sorted({c["file"] for c in dq.store.chunks})
    st.write(f"**Indexed:** {len(files)} file(s), {len(dq.store.chunks)} chunks")
    for f in files:
        st.write(f"- {f}")

question = st.text_input("Your question")
if question:
    if not dq.store.chunks:
        st.warning("Upload and index at least one PDF first.")
    else:
        with st.spinner("Searching and answering..."):
            res = dq.ask(question)
        if res["answerable"]:
            st.markdown(res["answer"])
            st.subheader("Sources")
            for s in res["sources"]:
                st.write(f"📄 **{s['file']}**, page {s['page']}")
        else:
            st.info(res["answer"])
            if res["note"]:
                st.caption(res["note"])
        with st.expander("Retrieved passages (what the model was given)"):
            for i, h in enumerate(res["hits"], 1):
                c = h["chunk"]
                st.markdown(f"**[{i}] {c['file']}, p.{c['page']}**")
                st.text(c["text"])
