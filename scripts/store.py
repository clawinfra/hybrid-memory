"""HybridStore — main facade for hybrid search memory."""
import logging
import os
import sqlite3

try:
    from .chunker import chunk_text
    from .embedder import OllamaEmbedder
    from .fts import FTSIndex
    from .vector import VectorSearch, encode_embedding
    from .merger import merge_results
except ImportError:
    from chunker import chunk_text
    from embedder import OllamaEmbedder
    from fts import FTSIndex
    from vector import VectorSearch, encode_embedding
    from merger import merge_results

log = logging.getLogger(__name__)

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS chunks (
    doc_id    TEXT NOT NULL,
    chunk_id  TEXT NOT NULL,
    heading   TEXT NOT NULL DEFAULT '',
    content   TEXT NOT NULL,
    embedding BLOB,
    created_at REAL NOT NULL DEFAULT (unixepoch('now')),
    PRIMARY KEY (doc_id, chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    doc_id,
    chunk_id,
    heading,
    content,
    tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id     TEXT PRIMARY KEY,
    source_path TEXT,
    file_mtime REAL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    indexed_at REAL NOT NULL DEFAULT (unixepoch('now'))
);
"""


class HybridStore:
    def __init__(self, db_path: str = "hybrid_search.db", config: dict | None = None) -> None:
        config = config or {}
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

        self.embedder = OllamaEmbedder(
            base_url=config.get("ollama_url", "http://localhost:11434"),
            model=config.get("model", "nomic-embed-text"),
            cache_size=config.get("cache_size", 256),
        )
        self.chunk_size = config.get("chunk_size", 512)
        self.chunk_overlap = config.get("chunk_overlap", 50)
        self.keyword_weight = config.get("keyword_weight", 0.3)
        self.vector_weight = config.get("vector_weight", 0.7)

        self.fts = FTSIndex(self.conn)
        self.vector = VectorSearch(self.conn, self.embedder)

        if not self.embedder.is_available():
            log.warning("Ollama not available — vector search disabled, keyword-only mode")

    def _init_schema(self) -> None:
        for stmt in SCHEMA.split(";"):
            stmt = stmt.strip()
            if stmt:
                try:
                    self.conn.execute(stmt)
                except sqlite3.OperationalError:
                    pass  # table already exists etc.
        self.conn.commit()

    def store(self, doc_id: str, text: str, metadata: dict[str, str] | None = None) -> int:
        """Chunk, embed, and index a document. Returns number of chunks stored."""
        # Delete existing chunks for this doc
        self.delete(doc_id)

        chunks = chunk_text(text, chunk_size=self.chunk_size, overlap=self.chunk_overlap)
        for chunk in chunks:
            chunk_id = f"{doc_id}-{chunk.index}"
            embedding = self.embedder.embed(chunk.text)
            emb_blob = encode_embedding(embedding) if embedding else None

            self.conn.execute(
                "INSERT INTO chunks (doc_id, chunk_id, heading, content, embedding) VALUES (?, ?, ?, ?, ?)",
                (doc_id, chunk_id, chunk.heading, chunk.text, emb_blob),
            )
            self.conn.execute(
                "INSERT INTO chunks_fts (doc_id, chunk_id, heading, content) VALUES (?, ?, ?, ?)",
                (doc_id, chunk_id, chunk.heading, chunk.text),
            )

        # Update documents table
        source_path = (metadata or {}).get("source_path", "")
        file_mtime = float((metadata or {}).get("file_mtime", 0))
        self.conn.execute(
            "INSERT OR REPLACE INTO documents (doc_id, source_path, file_mtime, chunk_count) VALUES (?, ?, ?, ?)",
            (doc_id, source_path, file_mtime, len(chunks)),
        )
        self.conn.commit()
        return len(chunks)

    def search(self, query: str, limit: int = 10, mode: str = "hybrid") -> list[dict]:
        """Search. mode: 'hybrid'|'keyword'|'vector'."""
        if mode == "keyword":
            return self.fts.search(query, limit)
        elif mode == "vector":
            return self.vector.search(query, limit)
        else:
            kw = self.fts.search(query, limit * 2)
            vec = self.vector.search(query, limit * 2)
            merged = merge_results(kw, vec, self.keyword_weight, self.vector_weight)
            return merged[:limit]

    def delete(self, doc_id: str) -> int:
        """Delete all chunks for a doc_id. Returns chunks deleted."""
        count = self.conn.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,)).fetchone()[0]
        self.conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        self.conn.execute("DELETE FROM chunks_fts WHERE doc_id = ?", (doc_id,))
        self.conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        self.conn.commit()
        return count

    def reindex(self) -> None:
        """Rebuild FTS5 index from chunks table."""
        self.fts.reindex()

    def stats(self) -> dict:
        """Return index statistics."""
        doc_count = self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunk_count = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        embedded_count = self.conn.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL").fetchone()[0]
        db_size = os.path.getsize(self.db_path) if self.db_path != ":memory:" and os.path.exists(self.db_path) else 0
        return {
            "doc_count": doc_count,
            "chunk_count": chunk_count,
            "embedded_count": embedded_count,
            "db_size_bytes": db_size,
        }

    def close(self) -> None:
        self.conn.close()
