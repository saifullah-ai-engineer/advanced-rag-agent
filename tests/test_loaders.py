"""Tests for file loading, including PDF, and embedder fallbacks."""

from advanced_rag_agent.config import RagConfig
from advanced_rag_agent.embeddings import HashEmbedder, get_embedder
from advanced_rag_agent.loaders import iter_files, load_file
from advanced_rag_agent.reranker import LexicalReranker, get_reranker


def _make_pdf_bytes(text: str) -> bytes:
    """Build a minimal one-page PDF with the given text, valid xref table."""
    content = f"BT /F1 24 Tf 100 700 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF".encode()
    )
    return out


def test_pdf_loader_extracts_text(tmp_path, config):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(_make_pdf_bytes("Hello PDF world"))
    chunks = load_file(pdf_path, config)
    assert len(chunks) >= 1
    assert "Hello PDF world" in chunks[0].text
    assert chunks[0].heading == "page 1"


def test_iter_files_finds_supported_types(tmp_path):
    (tmp_path / "a.md").write_text("# Hi\n")
    (tmp_path / "b.py").write_text("x = 1\n")
    (tmp_path / "c.txt").write_text("hello\n")
    (tmp_path / "d.png").write_bytes(b"not a document")
    found = iter_files(tmp_path)
    assert {p.name for p in found} == {"a.md", "b.py", "c.txt"}


def test_get_embedder_falls_back_to_hash_without_ml(capsys):
    config = RagConfig()
    embedder = get_embedder(config, prefer_real=True)
    assert isinstance(embedder, HashEmbedder)
    assert "Falling back" in capsys.readouterr().err


def test_get_reranker_falls_back_without_ml(config, capsys):
    reranker = get_reranker(config, prefer_cross=True)
    assert isinstance(reranker, LexicalReranker)
