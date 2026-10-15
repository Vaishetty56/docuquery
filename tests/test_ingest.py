"""Phase 2 tests: chunking rules and PDF reading."""
import pytest

from conftest import make_pdf
from docuquery.ingest import chunk_pages, chunk_text, clean_text, ingest_pdf

LONG = ("Risk management is a process. " * 80).strip()


def test_chunks_respect_size_limit():
    assert all(len(c) <= 1000 for c in chunk_text(LONG, 1000, 200))


def test_consecutive_chunks_overlap():
    chunks = chunk_text(LONG, 500, 100)
    assert len(chunks) > 2
    # the tail of one chunk reappears at the start of the next
    assert chunks[0][-50:].split(" ")[-2] in chunks[1]


def test_short_text_gives_one_chunk():
    assert chunk_text("A short but long enough sentence to be kept as is.", 1000, 200) == [
        "A short but long enough sentence to be kept as is."]


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("x" * 100, 50, 50)


def test_terminates_on_text_without_spaces():
    assert len(chunk_text("a" * 5000, 1000, 200)) >= 5


def test_clean_text_rejoins_hyphenated_words():
    assert clean_text("trust-\nworthy   AI\nsystems") == "trustworthy AI systems"


def test_chunks_never_cross_pages_and_keep_page_numbers():
    pages = [(1, LONG), (2, "Second page text that is definitely long enough to keep.")]
    chunks = chunk_pages(pages, "doc.pdf", 500, 100)
    assert {c["page"] for c in chunks} == {1, 2}
    assert all(c["file"] == "doc.pdf" for c in chunks)


def test_ingest_real_pdf(tmp_path):
    pdf = make_pdf(tmp_path / "t.pdf", ["Alpha page about governance of systems and policies.",
                                         "Beta page about measurement of risk and impacts."])
    chunks, empty = ingest_pdf(pdf, 1000, 200)
    assert empty == []
    assert [(c["page"], "governance" in c["text"]) for c in chunks] == [(1, True), (2, False)]
    assert chunks[0]["file"] == "t.pdf"
