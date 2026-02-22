# Hybrid Search Memory Skill — Technical Plan

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────┐
│                  hybrid_memory_cli.py            │
│              (CLI entry point, argparse)         │
└──────────┬──────────────────────────────────────┘
           │
┌──────────▼──────────────────────────────────────┐
│                    store.py                      │
│         (HybridStore — main facade)              │
│  store(doc_id, text, metadata) → chunks+embeds   │
│  search(query, limit) → merged results           │
│  delete(doc_id) → remove doc                     │
│  reindex() → rebuild FTS from chunks table       │
└──┬─────────┬──────────┬────────────┬────────────┘
   │         │          │            │
┌──▼──┐  ┌──▼───┐  ┌───▼────┐  ┌───▼─────┐
│chunk│  │ fts  │  │ vector │  │ merger  │
│er.py│  │.py   │  │.py     │  │.py      │
└─────┘  └──────┘  └────────┘  └─────────┘
                       │
                ┌──────▼──────┐
                │ embedder.py │
                │ (Ollama API)│
                └─────────────┘
```

**Data flow — Ingest:**
1. CLI receives `ingest --path memory/` or `store --doc-id X --text "..."`
2. `store.py` calls `chunker.py` → splits text into overlapping chunks with heading context
3. For each chunk: `embedder.py` calls Ollama `nomic-embed-text` → 768-dim float32 vector
4. Insert chunk text into `chunks` table + `chunks_fts` FTS5 virtual table
5. Store embedding as BLOB in `chunks.embedding` column

**Data flow — Search:**
1. CLI receives `search --query "ssh password" --limit 5`
2. `fts.py` runs BM25 keyword search → returns scored results
3. `vector.py` embeds query via Ollama, scans all embeddings for cosine similarity → returns scored results
4. `merger.py` normalizes both score sets, combines with weights (keyword=0.3, vector=0.7), deduplicates by chunk_id → returns ranked results

## 2. File Structure

```
skills/hybrid-memory/
├── PLAN.md                  # This file
├── SKILL.md                 # Skill documentation for agents
├── config.json              # Default configuration
├── scripts/
│   ├── hybrid_memory_cli.py # CLI entry point
│   ├── store.py             # HybridStore facade
│   ├── chunker.py           # Markdown-aware text chunking
│   ├── fts.py               # FTS5 keyword search
│   ├── vector.py            # Vector search + cosine similarity
│   ├── merger.py            # Result merging + normalization
│   └── embedder.py          # Ollama embedding client
└── tests/
    ├── test_store.py
    ├── test_chunker.py
    ├── test_fts.py
    ├── test_vector.py
    ├── test_merger.py
    ├── test_embedder.py
    └── test_cli.py
```

### Public API (function signatures)

#### `store.py`

```python
class HybridStore:
    def __init__(self, db_path: str = "hybrid_search.db", config: dict | None = None) -> None:
        """Open/create SQLite DB, run migrations, init components."""

    def store(self, doc_id: str, text: str, metadata: dict[str, str] | None = None) -> int:
        """Chunk, embed, and index a document. Returns number of chunks stored."""

    def search(self, query: str, limit: int = 10, mode: str = "hybrid") -> list[dict]:
        """Search. mode: 'hybrid'|'keyword'|'vector'. Returns list of SearchResult dicts."""

    def delete(self, doc_id: str) -> int:
        """Delete all chunks for a doc_id. Returns chunks deleted."""

    def reindex(self) -> None:
        """Rebuild FTS5 index from chunks table."""

    def stats(self) -> dict:
        """Return {doc_count, chunk_count, embedded_count, db_size_bytes}."""

    def close(self) -> None:
        """Close DB connection."""
```

Each `SearchResult` dict:
```python
{
    "doc_id": str,
    "chunk_id": str,
    "heading": str,
    "text": str,
    "score": float,       # 0.0–1.0 normalized
    "source": str,        # "keyword" | "vector" | "hybrid"
}
```

#### `chunker.py`

```python
@dataclass
class Chunk:
    text: str
    heading: str
    index: int

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[Chunk]:
    """Split markdown text into overlapping chunks preserving heading context."""
```

#### `fts.py`

```python
class FTSIndex:
    def __init__(self, conn: sqlite3.Connection) -> None: ...
    def search(self, query: str, limit: int = 10) -> list[dict]: ...
    def snippet(self, query: str, limit: int = 10) -> list[dict]: ...
    def reindex(self) -> None: ...
```

#### `vector.py`

```python
def cosine_similarity(a: list[float], b: list[float]) -> float: ...
def encode_embedding(v: list[float]) -> bytes: ...
def decode_embedding(b: bytes) -> list[float]: ...

class VectorSearch:
    def __init__(self, conn: sqlite3.Connection, embedder: "OllamaEmbedder") -> None: ...
    def search(self, query: str, limit: int = 10) -> list[dict]: ...
```

#### `merger.py`

```python
def merge_results(
    keyword_results: list[dict],
    vector_results: list[dict],
    keyword_weight: float = 0.3,
    vector_weight: float = 0.7,
) -> list[dict]:
    """Normalize, weight, deduplicate, sort by combined score."""
```

#### `embedder.py`

```python
class OllamaEmbedder:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text", cache_size: int = 256) -> None: ...
    def embed(self, text: str) -> list[float]:
        """Return 768-dim embedding. Uses LRU cache. Returns [] on failure."""
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Calls Ollama sequentially (no batch API)."""
    def dims(self) -> int:
        """Return 768."""
    def is_available(self) -> bool:
        """Check if Ollama is reachable and model loaded."""
```

## 3. Database Schema

Single SQLite file, WAL mode.

```sql
PRAGMA journal_mode=WAL;

-- Main chunks table
CREATE TABLE IF NOT EXISTS chunks (
    doc_id    TEXT NOT NULL,
    chunk_id  TEXT NOT NULL,
    heading   TEXT NOT NULL DEFAULT '',
    content   TEXT NOT NULL,
    embedding BLOB,            -- 768 × 4 bytes = 3072 bytes per chunk (float32)
    created_at REAL NOT NULL DEFAULT (unixepoch('now')),
    PRIMARY KEY (doc_id, chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

-- FTS5 virtual table (standalone, not content-synced)
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    doc_id,
    chunk_id,
    heading,
    content,
    tokenize='porter unicode61'
);

-- Document metadata (tracks source files for incremental re-indexing)
CREATE TABLE IF NOT EXISTS documents (
    doc_id     TEXT PRIMARY KEY,
    source_path TEXT,
    file_mtime REAL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    indexed_at REAL NOT NULL DEFAULT (unixepoch('now'))
);
```

**Embedding encoding:** `struct.pack(f'<{n}f', *embedding)` — little-endian float32. 768 dims × 4 bytes = 3072 bytes per chunk.

**Why float32 not float64:** Halves storage. nomic-embed-text precision doesn't benefit from float64.

## 4. Embedding Pipeline

### Ollama API call

```python
# POST http://localhost:11434/api/embed
# Body: {"model": "nomic-embed-text", "input": "text to embed"}
# Response: {"embeddings": [[0.1, 0.2, ...]]}
```

Use `requests.post()` with 30s timeout.

### Batching strategy

Ollama's `/api/embed` accepts a single `input` string (not batch). Strategy:
- Process chunks sequentially during ingest
- No parallelism needed — Ollama saturates GPU on single requests
- For bulk ingest (initial indexing), process one doc at a time, all chunks sequentially

### Caching

- In-memory LRU cache, keyed by text content (hash), capacity=256 entries
- Avoids re-embedding identical chunks during re-indexing
- Use `collections.OrderedDict` for LRU implementation
- Cache is per-process only (not persisted)

### Fallback behavior

1. If Ollama unreachable → log warning, store chunk with `embedding=NULL`, keyword-only search works
2. If embedding returns error → same: NULL embedding, skip vector search for that chunk
3. `search()` gracefully degrades: if no embeddings exist, returns keyword-only results
4. `is_available()` check at startup, log once if Ollama down

## 5. Search Algorithm

### Step 1: Keyword search (FTS5 BM25)

```sql
SELECT doc_id, chunk_id, heading, content, bm25(chunks_fts) AS score
FROM chunks_fts
WHERE chunks_fts MATCH ?
ORDER BY bm25(chunks_fts)  -- BM25 returns negative (lower=better)
LIMIT ?
```

Post-process: negate BM25 scores (`score = -bm25_score`) so higher = better.

FTS5 MATCH query: pass the raw query string. If it contains special FTS5 characters that cause errors, catch the exception and return empty results (graceful degradation).

### Step 2: Vector search (cosine similarity)

1. Embed query via `OllamaEmbedder.embed(query)`. If fails → return [].
2. Load ALL chunks with non-NULL embeddings:
   ```sql
   SELECT doc_id, chunk_id, heading, content, embedding FROM chunks WHERE embedding IS NOT NULL
   ```
3. For each row, decode embedding and compute cosine similarity.
4. Filter results with score > 0.0.
5. Sort descending, take top `limit`.

**Cosine similarity formula:**
```
sim(a, b) = (a · b) / (‖a‖ × ‖b‖)
```
Where `a · b = Σ(a_i × b_i)`, `‖a‖ = √(Σ(a_i²))`

**Performance note:** Full scan is fine for <100K chunks. For OpenClaw memory files (~100 docs, ~1000 chunks), this takes <10ms.

### Step 3: Merge (weighted combination)

1. **Normalize** each result set independently:
   - Find max score in set
   - Divide all scores by max → scores in [0, 1]
   - If max ≤ 0, skip normalization

2. **Combine** by chunk key (`doc_id:chunk_id`):
   - If chunk appears in keyword results only: `final_score = keyword_score × keyword_weight`
   - If chunk appears in vector results only: `final_score = vector_score × vector_weight`
   - If chunk appears in both: `final_score = keyword_score × keyword_weight + vector_score × vector_weight`, source = "hybrid"

3. **Sort** by final_score descending.

**Default weights:** keyword_weight=0.3, vector_weight=0.7 (matches Go implementation)

## 6. Chunking Algorithm

Port of Go `ChunkText()`:

1. Split text into lines by `\n`
2. Track `current_heading` (most recent line starting with `#`)
3. Accumulate lines into a buffer
4. When a heading line is encountered:
   - Flush current buffer as a chunk (if non-empty after strip)
   - Update `current_heading`
   - Start new buffer with the heading line
5. When buffer length + next line > `chunk_size` and buffer non-empty:
   - Flush buffer as chunk
   - Apply overlap: keep last `overlap` characters from previous buffer as prefix of new buffer
6. At end, flush remaining buffer

**Defaults:** chunk_size=512 chars, overlap=50 chars

**chunk_id format:** `{doc_id}-{index}` where index is 0-based sequential

## 7. CLI Interface

Entry point: `uv run python skills/hybrid-memory/scripts/hybrid_memory_cli.py <command>`

### Commands

```
hybrid_memory_cli.py store --doc-id ID --text TEXT [--file PATH] [--metadata KEY=VAL...]
    Store a document. --file reads from file path. --text reads inline.
    Output: JSON {"doc_id": "...", "chunks": N}

hybrid_memory_cli.py search --query QUERY [--limit 10] [--mode hybrid|keyword|vector]
    Search across indexed documents.
    Output: JSON array of SearchResult dicts

hybrid_memory_cli.py ingest --path DIR [--pattern "*.md"] [--incremental]
    Bulk ingest files from directory. --incremental skips unchanged files (by mtime).
    Output: JSON {"files": N, "chunks": N, "skipped": N, "errors": [...]}

hybrid_memory_cli.py delete --doc-id ID
    Delete a document and its chunks.
    Output: JSON {"doc_id": "...", "chunks_deleted": N}

hybrid_memory_cli.py reindex
    Rebuild FTS5 index from chunks table.
    Output: JSON {"status": "ok", "chunks": N}

hybrid_memory_cli.py stats
    Show index statistics.
    Output: JSON {"doc_count": N, "chunk_count": N, "embedded_count": N, "db_size_bytes": N}
```

### Global options

```
--db PATH        SQLite database path (default: skills/hybrid-memory/data/hybrid_search.db)
--ollama-url URL Ollama base URL (default: http://localhost:11434)
--model MODEL    Embedding model (default: nomic-embed-text)
--chunk-size N   Chunk size in chars (default: 512)
--overlap N      Chunk overlap in chars (default: 50)
--keyword-weight F  Keyword weight (default: 0.3)
--vector-weight F   Vector weight (default: 0.7)
--quiet          Suppress stderr warnings
```

### Config file

`config.json` provides defaults (CLI args override):

```json
{
  "db_path": "data/hybrid_search.db",
  "ollama_url": "http://localhost:11434",
  "model": "nomic-embed-text",
  "chunk_size": 512,
  "chunk_overlap": 50,
  "keyword_weight": 0.3,
  "vector_weight": 0.7,
  "cache_size": 256,
  "dims": 768
}
```

## 8. Integration Points

### With tiered memory

These are **complementary, not competing** systems:
- **Tiered memory** = LLM-powered tree navigation (semantic understanding, O(log n))
- **Hybrid search** = keyword + vector (fast exact matches, similarity search)

Agent workflow in AGENTS.md should be updated:
```
Before answering questions:
1. Search hybrid memory: `uv run python skills/hybrid-memory/scripts/hybrid_memory_cli.py search --query "..." --limit 5`
2. If results insufficient, fall back to tiered memory: `python3 skills/tiered-memory/scripts/memory_cli.py retrieve "..." --limit 5`
```

### Ingest from memory files

The `ingest` command handles bulk indexing:
```bash
# Initial dogfooding — index all memory markdown
uv run python skills/hybrid-memory/scripts/hybrid_memory_cli.py ingest \
  --path memory/ --pattern "*.md"

# Also index top-level docs
uv run python skills/hybrid-memory/scripts/hybrid_memory_cli.py store \
  --doc-id TOOLS --file TOOLS.md
uv run python skills/hybrid-memory/scripts/hybrid_memory_cli.py store \
  --doc-id MEMORY --file MEMORY.md
```

### HEARTBEAT.md integration

Add to HEARTBEAT.md periodic task:
```markdown
## Hybrid Memory Reindex (every 6h)
- Run: `uv run python skills/hybrid-memory/scripts/hybrid_memory_cli.py ingest --path memory/ --pattern "*.md" --incremental`
- Only re-indexes files with changed mtime (fast)
```

### Incremental indexing logic

The `documents` table tracks `file_mtime`. During `ingest --incremental`:
1. `os.stat(path).st_mtime` for each file
2. Compare with stored `file_mtime` in `documents` table
3. Skip if unchanged, re-index if modified or new

## 9. Test Plan

All tests use `":memory:"` SQLite database and a mock embedder.

### Mock Ollama

```python
class MockEmbedder:
    """Returns deterministic embeddings based on text hash."""
    def embed(self, text: str) -> list[float]:
        h = hashlib.md5(text.encode()).digest()
        # Expand 16 bytes to 768 floats deterministically
        rng = random.Random(int.from_bytes(h, 'big'))
        return [rng.gauss(0, 1) for _ in range(768)]
    def embed_batch(self, texts): return [self.embed(t) for t in texts]
    def dims(self): return 768
    def is_available(self): return True
```

### Test cases

**test_chunker.py:**
- Empty text → []
- Single line → 1 chunk
- Text with headings → chunks split at headings, heading preserved
- Chunk exceeds chunk_size → splits with overlap
- Overlap carries correct suffix
- Unicode text handling

**test_fts.py:**
- Index + search returns BM25-ranked results
- Search for non-existent term → []
- Special characters in query → graceful failure (empty results, no crash)
- Snippet returns highlighted text
- Reindex rebuilds correctly after delete

**test_vector.py:**
- cosine_similarity([1,0],[0,1]) == 0.0
- cosine_similarity([1,0],[1,0]) == 1.0
- cosine_similarity with zero vector → 0.0
- encode_embedding/decode_embedding roundtrip
- VectorSearch with mock embedder returns ranked results
- VectorSearch with no embeddings → []

**test_merger.py:**
- Merge keyword-only results
- Merge vector-only results
- Merge overlapping results → deduplication, "hybrid" source label
- Normalization with all-zero scores
- Weight=0 for one side → other side dominates
- Empty inputs → []

**test_store.py:**
- store() then search() finds document
- store() same doc_id twice → replaces (upsert)
- delete() removes all chunks
- search with mode="keyword" → keyword-only
- search with mode="vector" → vector-only
- Graceful degradation: embedder fails → keyword-only still works
- stats() returns correct counts
- reindex() rebuilds FTS

**test_embedder.py:**
- Mock HTTP server for Ollama API
- embed() returns 768-dim vector
- embed() with unreachable server → returns []
- LRU cache hit (same text, no HTTP call)
- LRU cache eviction (exceed capacity)
- is_available() with/without server

**test_cli.py:**
- End-to-end: `store` then `search` via subprocess
- `ingest` with temp directory of .md files
- `ingest --incremental` skips unchanged files
- `delete` removes document
- `stats` returns valid JSON
- Invalid args → non-zero exit code

### Coverage target: >85%

## 10. Constraints

- **Dependencies:** stdlib + `requests` only. No numpy, no faiss, no sentence-transformers.
- **Runner:** `uv run python` (uv handles venv/deps automatically)
- **Python:** 3.11+ (for `sqlite3` with FTS5 support, which is built-in on modern Python)
- **Graceful degradation:** If Ollama is down, keyword search still works. Never crash on embedding failure.
- **No CGO:** Pure Python sqlite3 stdlib module (FTS5 available since Python 3.7+ with system SQLite ≥3.9)
- **Thread safety:** Not required (CLI is single-process). But DB uses WAL for potential future concurrency.
- **Embedding storage:** float32 (not float64) to halve BLOB size. 768×4=3072 bytes/chunk.
- **Max practical corpus:** ~10K documents, ~100K chunks. Full vector scan is fine at this scale.

## 11. Dogfooding Plan

### Step 1: Build and verify (Builder task)

Implement all files per this plan. Run tests. Verify >85% coverage.

### Step 2: Initial indexing

```bash
# Create data directory
mkdir -p skills/hybrid-memory/data

# Index all memory markdown files
uv run python skills/hybrid-memory/scripts/hybrid_memory_cli.py ingest \
  --path memory/ --pattern "*.md"

# Index key workspace docs
for f in TOOLS.md MEMORY.md AGENTS.md; do
  uv run python skills/hybrid-memory/scripts/hybrid_memory_cli.py store \
    --doc-id "$f" --file "$f"
done

# Verify
uv run python skills/hybrid-memory/scripts/hybrid_memory_cli.py stats
```

### Step 3: Test credential/config lookups

```bash
# Should find GPU server password from TOOLS.md
uv run python skills/hybrid-memory/scripts/hybrid_memory_cli.py search \
  --query "GPU server SSH password" --limit 3

# Should find Pi IP from TOOLS.md
uv run python skills/hybrid-memory/scripts/hybrid_memory_cli.py search \
  --query "Raspberry Pi IP address" --limit 3

# Should find Ollama config
uv run python skills/hybrid-memory/scripts/hybrid_memory_cli.py search \
  --query "Ollama embedding model" --limit 3
```

### Step 4: Wire into agent workflow

Update `AGENTS.md` Memory Retrieval Protocol to add hybrid search as first-pass:

```markdown
### Standard Retrieval Flow:
1. **Hybrid search** (fast, precise):
   ```bash
   uv run python skills/hybrid-memory/scripts/hybrid_memory_cli.py search --query "..." --limit 5
   ```
2. **If insufficient**, use tiered memory (semantic, LLM-powered):
   ```bash
   python3 skills/tiered-memory/scripts/memory_cli.py retrieve "..." --limit 5
   ```
```

### Step 5: Add to HEARTBEAT.md

```markdown
## Hybrid Memory Maintenance (every 6h)
uv run python skills/hybrid-memory/scripts/hybrid_memory_cli.py ingest --path memory/ --pattern "*.md" --incremental
```

### Step 6: Create SKILL.md

Write `skills/hybrid-memory/SKILL.md` documenting:
- What it does
- CLI usage examples
- When to use hybrid vs tiered memory
- Configuration options
