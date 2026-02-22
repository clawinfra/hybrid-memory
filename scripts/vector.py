"""Vector search with cosine similarity."""
import math
import struct
import sqlite3
import logging

log = logging.getLogger(__name__)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def encode_embedding(v: list[float]) -> bytes:
    """Encode embedding as little-endian float32 bytes."""
    return struct.pack(f"<{len(v)}f", *v)


def decode_embedding(b: bytes) -> list[float]:
    """Decode little-endian float32 bytes to embedding."""
    n = len(b) // 4
    return list(struct.unpack(f"<{n}f", b))


class VectorSearch:
    def __init__(self, conn: sqlite3.Connection, embedder) -> None:
        self.conn = conn
        self.embedder = embedder

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Embed query and find most similar chunks by cosine similarity."""
        query_vec = self.embedder.embed(query)
        if not query_vec:
            return []

        rows = self.conn.execute(
            "SELECT doc_id, chunk_id, heading, content, embedding FROM chunks WHERE embedding IS NOT NULL"
        ).fetchall()

        results = []
        for r in rows:
            chunk_vec = decode_embedding(r[4])
            sim = cosine_similarity(query_vec, chunk_vec)
            if sim > 0.0:
                results.append({
                    "doc_id": r[0],
                    "chunk_id": r[1],
                    "heading": r[2],
                    "text": r[3],
                    "score": sim,
                    "source": "vector",
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
