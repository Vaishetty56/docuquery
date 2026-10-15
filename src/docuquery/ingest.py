"""Phase 2: PDF -> pages -> chunks.

Key design choice: a chunk NEVER crosses a page boundary. That way every chunk
has exactly one (file, page) and citations are exact. The cost: an idea that
spans two pages is split, which retrieval has to cope with (we test this on purpose).
"""
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

MIN_CHUNK_CHARS = 40  # drop fragments like page numbers / running headers


def clean_text(text: str) -> str:
    """Fix common PDF extraction noise."""
    # NFKC turns PDF ligatures (the single character "fi", "fl", ...) into plain letters,
    # so keyword search can match words like "financial".
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("-\n", "")          # re-join words hyphenated at line ends
    text = re.sub(r"\s+", " ", text)        # collapse newlines/tabs/multiple spaces
    return text.strip()


def extract_pages(pdf_path) -> list[tuple[int, str]]:
    """Return [(page_number, text)] with 1-based PHYSICAL page numbers
    (the n-th page in the file, which is what a PDF viewer shows in its page box;
    printed page labels inside a book can differ)."""
    reader = PdfReader(str(pdf_path))
    return [(i + 1, clean_text(page.extract_text() or "")) for i, page in enumerate(reader.pages)]


def chunk_text(text: str, size: int = 1000, overlap: int = 200) -> list[str]:
    """Split text into windows of ~`size` characters sharing `overlap` characters.

    We try to end each chunk at a sentence end (". ") or at least a word boundary,
    searching only the last 20% of the window so chunks stay near `size`.
    """
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    text = text.strip()
    n = len(text)
    chunks, start = [], 0
    while start < n:
        end = min(start + size, n)
        if end < n:  # not the last window: look for a clean place to cut
            lo = start + int(size * 0.8)
            cut = text.rfind(". ", lo, end)
            if cut == -1:
                cut = text.rfind(" ", lo, end)
            if cut != -1:
                end = cut + 1
        piece = text[start:end].strip()
        if len(piece) >= MIN_CHUNK_CHARS:
            chunks.append(piece)
        if end >= n:
            break
        nxt = end - overlap
        if nxt <= start:          # guarantee progress
            nxt = end
        else:                     # start the next chunk at a word boundary
            space = text.find(" ", nxt, end)
            nxt = space + 1 if space != -1 else end
        start = nxt
    return chunks


def chunk_pages(pages, file_name: str, size: int, overlap: int) -> list[dict]:
    """Turn [(page, text)] into chunk dicts: {file, page, text}."""
    out = []
    for page_no, text in pages:
        for piece in chunk_text(text, size, overlap):
            out.append({"file": file_name, "page": page_no, "text": piece})
    return out


def ingest_pdf(pdf_path, size: int = 1000, overlap: int = 200):
    """Read one PDF. Returns (chunks, empty_pages).

    empty_pages lists pages with no extractable text (usually scanned images;
    we do not OCR, so those pages are invisible to search. We surface this
    instead of failing silently).
    """
    pdf_path = Path(pdf_path)
    pages = extract_pages(pdf_path)
    empty = [p for p, t in pages if not t]
    return chunk_pages(pages, pdf_path.name, size, overlap), empty