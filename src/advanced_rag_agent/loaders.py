"""File loaders for the ingestion lane.

Supported inputs: Markdown, plain text, PDF, and Python source files.
Every loader returns Chunk objects, so the rest of the pipeline never has
to care which file type a chunk came from.
"""

from pathlib import Path

from advanced_rag_agent.chunking import Chunk, chunk_code, chunk_text
from advanced_rag_agent.config import RagConfig

TEXT_SUFFIXES = {".md", ".markdown", ".txt"}
CODE_SUFFIXES = {".py"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | CODE_SUFFIXES | {".pdf"}

SKIP_DIRS = {".venv", "__pycache__", ".git", "node_modules", ".tox", "dist", "build"}


def load_text_file(path: Path, config: RagConfig) -> list[Chunk]:
    """Load a Markdown or text file into document chunks."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return chunk_text(text, str(path), config.chunk_size, config.chunk_overlap)


def load_pdf_file(path: Path, config: RagConfig) -> list[Chunk]:
    """Load a PDF into document chunks, one heading per page."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    chunks: list[Chunk] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        page_chunks = chunk_text(text, str(path), config.chunk_size, config.chunk_overlap)
        for chunk in page_chunks:
            chunk.heading = f"page {page_number}"
            chunks.append(chunk)
    return chunks


def load_code_file(path: Path, config: RagConfig) -> list[Chunk]:
    """Load a code file into AST-aware chunks."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return chunk_code(
        text,
        str(path),
        chunk_size=config.chunk_size,
        window=config.code_window_lines,
        overlap=config.code_window_overlap,
    )


def load_file(path: str | Path, config: RagConfig) -> list[Chunk]:
    """Load one file by suffix. Raises ValueError for unsupported files."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return load_text_file(path, config)
    if suffix == ".pdf":
        return load_pdf_file(path, config)
    if suffix in CODE_SUFFIXES:
        return load_code_file(path, config)
    raise ValueError(f"Unsupported file type: {suffix} ({path})")


def iter_files(root: str | Path) -> list[Path]:
    """Walk a directory or single file and return supported files in order."""
    root = Path(root)
    if root.is_file():
        return [root] if root.suffix.lower() in SUPPORTED_SUFFIXES else []
    found: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            found.append(path)
    return found


def load_path(root: str | Path, config: RagConfig) -> list[Chunk]:
    """Load every supported file under a path into chunks."""
    chunks: list[Chunk] = []
    for path in iter_files(root):
        try:
            chunks.extend(load_file(path, config))
        except Exception as exc:  # noqa: BLE001 - report and keep going
            print(f"warning: skipped {path}: {exc}")
    return chunks
