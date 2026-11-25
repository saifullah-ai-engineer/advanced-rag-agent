"""Tests for document and code chunking."""

from advanced_rag_agent.chunking import chunk_code, chunk_text


def test_doc_chunking_splits_on_headings(doc_text, config):
    chunks = chunk_text(doc_text, "guide.md", config.chunk_size, config.chunk_overlap)
    headings = {c.heading for c in chunks}
    assert headings == {"First heading", "Second heading"}
    for chunk in chunks:
        assert chunk.kind == "docs"
        assert chunk.source == "guide.md"


def test_doc_chunking_overlap_present(config):
    paragraphs = ["word " * 12, "term " * 12, "text " * 12, "data " * 12]
    text = "\n\n".join(p.strip() for p in paragraphs)
    chunks = chunk_text(text, "doc.md", chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 2
    assert chunks[1].text.startswith(chunks[0].text[-20:])


def test_doc_chunking_small_text_single_chunk(config):
    chunks = chunk_text("short text", "doc.md", config.chunk_size, config.chunk_overlap)
    assert len(chunks) == 1
    assert chunks[0].text == "short text"


def test_doc_chunking_empty_text_returns_nothing(config):
    assert chunk_text("   ", "doc.md", config.chunk_size, config.chunk_overlap) == []


def test_code_chunking_splits_functions(code_text, config):
    chunks = chunk_code(code_text, "calc.py", chunk_size=config.chunk_size)
    names = {c.name for c in chunks}
    assert "add" in names
    assert "multiply" in names
    assert "Calculator" in names
    assert all(c.kind == "code" for c in chunks)


def test_code_chunking_class_kept_together(code_text, config):
    chunks = chunk_code(code_text, "calc.py", chunk_size=config.chunk_size)
    calculator = [c for c in chunks if c.name == "Calculator"]
    assert len(calculator) == 1
    assert "subtract" in calculator[0].text


def test_code_chunking_fallback_for_bad_syntax(config):
    bad_code = "def broken(:\n    pass\n" + "x = 1\n" * 80
    chunks = chunk_code(bad_code, "bad.py", chunk_size=config.chunk_size)
    assert len(chunks) >= 1
    assert all(c.kind == "code" for c in chunks)
