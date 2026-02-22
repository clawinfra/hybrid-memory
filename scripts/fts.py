"""FTS5 keyword search module."""
import logging
import sqlite3

log = logging.getLogger(__name__)


class FTSIndex:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Run BM25 keyword search. Returns results with positive scores (higher=better)."""
        if not query.strip():
            return []
        try:
            rows = self.conn.execute(
                "SELECT doc_id, chunk_id, heading, content, bm25(chunks_fts) AS score "
                "FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
                (query, limit),
            ).fetchall()
        except sqlite3.OperationalError as e:
            log.warning("FTS query failed: %s", e)
            return []

        return [
            {
                "doc_id": r[0],
                "chunk_id": r[1],
                "heading": r[2],
                "text": r[3],
                "score": -r[4],  # negate BM25 (lower=better → higher=better)
                "source": "keyword",
            }
            for r in rows
        ]

    def snippet(self, query: str, limit: int = 10) -> list[dict]:
        """Search with highlighted snippets."""
        if not query.strip():
            return []
        try:
            rows = self.conn.execute(
                "SELECT doc_id, chunk_id, heading, snippet(chunks_fts, 3, '**', '**', '...', 32), bm25(chunks_fts) "
                "FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
                (query, limit),
            ).fetchall()
        except sqlite3.OperationalError as e:
            log.warning("FTS snippet query failed: %s", e)
            return []

        return [
            {
                "doc_id": r[0],
                "chunk_id": r[1],
                "heading": r[2],
                "text": r[3],
                "score": -r[4],
                "source": "keyword",
            }
            for r in rows
        ]

    def reindex(self) -> None:
        """Rebuild FTS index from chunks table."""
        self.conn.execute("DELETE FROM chunks_fts")
        self.conn.execute(
            "INSERT INTO chunks_fts(doc_id, chunk_id, heading, content) "
            "SELECT doc_id, chunk_id, heading, content FROM chunks"
        )
        self.conn.commit()
