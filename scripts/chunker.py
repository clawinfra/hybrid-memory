"""Markdown-aware text chunking with heading context and overlap."""
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    heading: str
    index: int


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[Chunk]:
    """Split markdown text into overlapping chunks preserving heading context."""
    if not text or not text.strip():
        return []

    lines = text.split("\n")
    chunks: list[Chunk] = []
    current_heading = ""
    buffer = ""
    idx = 0

    def flush(buf: str):
        nonlocal idx
        stripped = buf.strip()
        if stripped:
            chunks.append(Chunk(text=stripped, heading=current_heading, index=idx))
            idx += 1

    for line in lines:
        is_heading = line.lstrip().startswith("#")

        if is_heading and buffer.strip():
            overlap_text = buffer[-overlap:] if overlap and len(buffer) > overlap else ""
            flush(buffer)
            current_heading = line.strip()
            buffer = overlap_text + line + "\n"
            continue

        if is_heading:
            current_heading = line.strip()

        candidate = buffer + line + "\n"
        if len(candidate) > chunk_size and buffer.strip():
            overlap_text = buffer[-overlap:] if overlap and len(buffer) > overlap else ""
            flush(buffer)
            buffer = overlap_text + line + "\n"
        else:
            buffer = candidate

        # Force-split if buffer alone exceeds chunk_size (long lines without newlines)
        while len(buffer) > chunk_size:
            piece = buffer[:chunk_size]
            remainder = buffer[chunk_size:]
            overlap_text = piece[-overlap:] if overlap and len(piece) > overlap else ""
            flush(piece)
            buffer = overlap_text + remainder

    flush(buffer)
    return chunks
