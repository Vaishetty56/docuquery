"""Phase 3 + 9 tests: vector store and retrieval variants (offline, with fakes)."""
import pytest

from conftest import FakeLLM
from docuquery.retrieval import Retriever, rrf_fuse
from docuquery.store import VectorStore

TEXTS = [
    "governance policies accountability structures for organizations",
    "measuring risk with quantitative metrics and benchmarks",
    "privacy enhancing technologies and data minimization practices",
    "zebra stripes camouflage predators unrelated filler passage",
]


@pytest.fixture
def store(embedder):
    s = VectorStore({"embed_model": "fake"})
    chunks = [{"file": "a.pdf", "page": i + 1, "text": t} for i, t in enumerate(TEXTS)]
    s.add(chunks, embedder.embed_documents(TEXTS))
    return s


def test_search_returns_most_similar_first(store, embedder):
    hits = store.search(embedder.embed_query("risk metrics benchmarks"), 2)
    assert hits[0][0]["page"] == 2
    assert hits[0][1] >= hits[1][1]


def test_k_larger_than_index_is_ok(store, embedder):
    assert len(store.search(embedder.embed_query("anything"), 50)) == 4


def test_save_load_roundtrip(store, embedder, tmp_path):
    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path)
    assert loaded.meta == {"embed_model": "fake"}
    assert [c["text"] for c in loaded.chunks] == TEXTS
    assert loaded.search(embedder.embed_query("privacy data"), 1)[0][0]["page"] == 3


def test_has_file(store):
    assert store.has_file("a.pdf") and not store.has_file("b.pdf")


def test_mismatched_lengths_rejected(store, embedder):
    with pytest.raises(ValueError):
        store.add([{"file": "x", "page": 1, "text": "t"}], embedder.embed_documents(["a", "b"]))


def test_rrf_rewards_agreement():
    # id 2 is ranked 2nd by both lists, id 1 first by one list only
    assert rrf_fuse([[1, 2, 3], [4, 2, 5]], 5)[0] == 2


def test_baseline_retriever(store, embedder):
    hits, info = Retriever(store, embedder, top_k=2).retrieve("risk metrics benchmarks")
    assert len(hits) == 2 and hits[0]["chunk"]["page"] == 2 and info["rerank_ok"] is None


def test_hybrid_finds_exact_keyword(store, embedder):
    hits, _ = Retriever(store, embedder, top_k=2, pool=4, hybrid=True).retrieve("zebra")
    assert hits[0]["chunk"]["page"] == 4


def test_rerank_reorders_candidates(store, embedder):
    llm = FakeLLM(lambda p, s, j: {"ranking": [3, 1, 2, 4]})
    hits, info = Retriever(store, embedder, llm, top_k=2, pool=4, rerank=True).retrieve("q")
    assert info["rerank_ok"] is True
    assert len(hits) == 2


def test_rerank_failure_is_reported_and_order_kept(store, embedder):
    llm = FakeLLM(lambda p, s, j: "not json at all")
    _, info = Retriever(store, embedder, llm, top_k=2, pool=4, rerank=True).retrieve("risk")
    assert info["rerank_ok"] is False


def test_rewrite_changes_query_used(store, embedder):
    llm = FakeLLM(lambda p, s, j: "privacy data minimization")
    hits, info = Retriever(store, embedder, llm, top_k=1, rewrite=True).retrieve("vague question")
    assert info["query_used"] == "privacy data minimization"
    assert hits[0]["chunk"]["page"] == 3


def test_rerank_without_llm_is_an_error(store, embedder):
    with pytest.raises(ValueError):
        Retriever(store, embedder, None, rerank=True)
