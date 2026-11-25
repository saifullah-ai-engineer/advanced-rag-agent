"""Chunking for documents and code.

Documents are split with a recursive splitter that is aware of markdown
headings: a chunk never mixes two sections, and each chunk remembers the
heading it came from. Code is split with the Python AST: top-level
functions and classes become chunks, and anything the parser cannot handle
falls back to a sliding line window.
"""

import ast
import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """One retrievable unit of text."""

    text: str
    source: str
    kind: str = "docs"
    name: str = ""
    heading: str = ""
    start_line: int = 1
    end_line: int = 1
    id: int | None = field(default=None, compare=False)

    def citation(self) -> str:
        """Short citation string used in answers, for example src/a.py:12-40."""
        return f"{self.source}:{self.start_line}-{self.end_line}"


def _split_long_piece(piece: str, chunk_size: int) -> list[str]:
    """Split one oversized piece, first on sentences, then on words."""
    if len(piece) <= chunk_size:
        return [piece]
    sentences = re.split(r"(?<=[.!?])\s+", piece)
    parts: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                parts.append(current)
            if len(sentence) > chunk_size:
                words = sentence.split()
                word_chunk = ""
                for word in words:
                    if len(word_chunk) + len(word) + 1 <= chunk_size:
                        word_chunk = f"{word_chunk} {word}".strip()
                    else:
                        parts.append(word_chunk)
                        word_chunk = word
                if word_chunk:
                    parts.append(word_chunk)
                current = ""
            else:
                current = sentence
    if current:
        parts.append(current)
    return [p for p in parts if p.strip()]


def _pack_segments(segments: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Greedily pack segments into chunks, carrying overlap between chunks."""
    chunks: list[str] = []
    current = ""
    for segment in segments:
        candidate = f"{current}\n\n{segment}".strip() if current else segment
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(segment) > chunk_size:
                for piece in _split_long_piece(segment, chunk_size):
                    chunks.append(piece)
                current = ""
            else:
                current = segment
    if current:
        chunks.append(current)
    if overlap <= 0 or len(chunks) < 2:
        return chunks
    overlapped: list[str] = [chunks[0]]
    for chunk in chunks[1:]:
        tail = overlapped[-1][-overlap:]
        overlapped.append(f"{tail}\n\n{chunk}" if len(chunk) + len(tail) + 2 <= chunk_size + overlap else chunk)
    return overlapped


def _split_markdown_sections(text: str) -> list[tuple[str, str, int]]:
    """Split markdown into (heading, section text, start line) blocks."""
    lines = text.splitlines()
    sections: list[tuple[str, str, int]] = []
    heading = ""
    start = 1
    buffer: list[str] = []
    for index, line in enumerate(lines, start=1):
        match = re.match(r"^#{1,6}\s+(.*)", line)
        if match and buffer:
            sections.append((heading, "\n".join(buffer), start))
            heading = match.group(1).strip()
            start = index
            buffer = [line]
        elif match:
            heading = match.group(1).strip()
            start = index
            buffer = [line]
        else:
            buffer.append(line)
    if buffer:
        sections.append((heading, "\n".join(buffer), start))
    return sections


def chunk_text(text: str, source: str, chunk_size: int = 800, chunk_overlap: int = 150) -> list[Chunk]:
    """Chunk document text, keeping markdown headings attached to chunks."""
    text = text.strip()
    if not text:
        return []
    chunks: list[Chunk] = []
    for heading, section, section_start in _split_markdown_sections(text):
        segments = [s.strip() for s in section.split("\n\n") if s.strip()]
        packed = _pack_segments(segments, chunk_size, chunk_overlap)
        cursor = 0
        for piece in packed:
            offset = section.find(piece[:60], cursor)
            if offset < 0:
                offset = cursor
            start_line = section_start + section[:offset].count("\n")
            end_line = start_line + piece.count("\n")
            chunks.append(
                Chunk(
                    text=piece,
                    source=source,
                    kind="docs",
                    heading=heading,
                    start_line=start_line,
                    end_line=end_line,
                )
            )
            cursor = offset + max(len(piece) - chunk_overlap, 1)
    return chunks


def _line_window_chunks(
    lines: list[str], source: str, window: int, overlap: int
) -> list[Chunk]:
    """Sliding window over lines, used for non-Python code and parse failures."""
    chunks: list[Chunk] = []
    step = max(window - overlap, 1)
    start = 0
    while start < len(lines):
        end = min(start + window, len(lines))
        text = "".join(lines[start:end]).strip()
        if text:
            chunks.append(
                Chunk(
                    text=text,
                    source=source,
                    kind="code",
                    start_line=start + 1,
                    end_line=end,
                )
            )
        if end >= len(lines):
            break
        start += step
    return chunks


def chunk_python_code(
    text: str, source: str, chunk_size: int = 800, window: int = 60, overlap: int = 10
) -> list[Chunk]:
    """Chunk Python source with the AST, one chunk per top-level definition."""
    lines = text.splitlines(keepends=True)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _line_window_chunks(lines, source, window, overlap)
    chunks: list[Chunk] = []
    covered: set[int] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            segment = ast.get_source_segment(text, node) or ""
            if not segment.strip():
                continue
            if isinstance(node, ast.ClassDef) and len(segment) > chunk_size:
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        child_segment = ast.get_source_segment(text, child) or ""
                        if child_segment.strip():
                            chunks.append(
                                Chunk(
                                    text=child_segment,
                                    source=source,
                                    kind="code",
                                    name=f"{node.name}.{child.name}",
                                    start_line=child.lineno,
                                    end_line=child.end_lineno or child.lineno,
                                )
                            )
                            covered.update(range(child.lineno, (child.end_lineno or child.lineno) + 1))
                continue
            chunks.append(
                Chunk(
                    text=segment,
                    source=source,
                    kind="code",
                    name=node.name,
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                )
            )
            covered.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    leftovers = [
        (i + 1, line) for i, line in enumerate(lines) if (i + 1) not in covered and line.strip()
    ]
    if leftovers:
        leftover_text = "".join(line for _, line in leftovers)
        if leftover_text.strip():
            chunks.append(
                Chunk(
                    text=leftover_text.strip(),
                    source=source,
                    kind="code",
                    name="module",
                    start_line=leftovers[0][0],
                    end_line=leftovers[-1][0],
                )
            )
    chunks.sort(key=lambda c: c.start_line)
    return chunks


def chunk_code(
    text: str, source: str, chunk_size: int = 800, window: int = 60, overlap: int = 10
) -> list[Chunk]:
    """Chunk a code file. Python gets AST chunks, other files get line windows."""
    if source.endswith(".py"):
        return chunk_python_code(text, source, chunk_size=chunk_size, window=window, overlap=overlap)
    lines = text.splitlines(keepends=True)
    return _line_window_chunks(lines, source, window, overlap)
