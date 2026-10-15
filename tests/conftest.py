"""Shared test helpers: fake embedder / LLM (no network, no API key) and a tiny PDF builder."""
import json
import zlib

import numpy as np
import pytest


class FakeEmbedder:
    """Hashes words into a 256-dim bag-of-words vector. Deterministic, offline."""
    def _vec(self, text):
        v = np.zeros(256, dtype="float32")
        for w in text.lower().split():
            v[zlib.crc32(w.strip(".,?!").encode()) % 256] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts):
        return np.array([self._vec(t) for t in texts], dtype="float32")

    def embed_query(self, text):
        return self._vec(text).reshape(1, -1)


class FakeLLM:
    """Returns whatever `responder(prompt, system, json_mode)` returns; records calls."""
    def __init__(self, responder):
        self.responder, self.calls = responder, []

    def generate(self, prompt, system=None, json_mode=False, temperature=0.0):
        self.calls.append(prompt)
        r = self.responder(prompt, system, json_mode)
        return r if isinstance(r, str) else json.dumps(r)


def make_pdf(path, pages_text):
    """Write a minimal valid PDF with one line of text per page (no extra libraries needed)."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{3 + 2*i} 0 R" for i in range(len(pages_text)))
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages_text)} >>")
    font_id = 3 + 2 * len(pages_text)
    for i, text in enumerate(pages_text):
        content_id = 4 + 2 * i
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                    f"/Contents {content_id} 0 R /Resources << /Font << /F1 {font_id} 0 R >> >> >>")
        stream = f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET"
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    objs.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out, offsets = b"%PDF-1.4\n", []
    for n, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs)+1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    path.write_bytes(out)
    return path


@pytest.fixture
def embedder():
    return FakeEmbedder()
