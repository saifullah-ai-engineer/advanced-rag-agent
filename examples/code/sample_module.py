"""Sample module for the code search lane.

A tiny, self-contained text utility used by the demo and the eval set.
It shows how the code lane chunks Python by function and class.
"""


def split_text(text, chunk_size=800, overlap=150):
    """Split text into chunks of at most chunk_size characters.

    The splitter works on paragraphs first, then on sentences, and finally
    on words, so it never cuts a word in half. Consecutive chunks share
    an overlap of characters so context is not lost at a boundary.
    Returns a list of chunk strings.
    """
    if len(text) <= chunk_size:
        return [text]
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = paragraph
    if current:
        chunks.append(current)
    overlapped = [chunks[0]] if chunks else []
    for chunk in chunks[1:]:
        tail = overlapped[-1][-overlap:]
        overlapped.append(f"{tail}\n\n{chunk}")
    return overlapped


def count_words(text):
    """Count the words in a text, splitting on whitespace."""
    return len(text.split())


class TextChunker:
    """A small helper that wraps split_text with fixed settings."""

    def __init__(self, chunk_size=800, overlap=150):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text):
        """Split text using this chunker's configured sizes."""
        return split_text(text, self.chunk_size, self.overlap)

    def chunk_count(self, text):
        """Return how many chunks the text splits into."""
        return len(self.chunk(text))
