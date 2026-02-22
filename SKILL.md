# Hybrid Search Memory

Keyword (BM25) + vector (cosine similarity) search over memory files and workspace documents.

## Usage

```bash
# Search (hybrid mode by default — falls back to keyword-only if Ollama is down)
uv run python skills/hybrid-memory/scripts/hybrid_cli.py search "query text"
uv run python skills/hybrid-memory/scripts/hybrid_cli.py search "query" --mode keyword
uv run python skills/hybrid-memory/scripts/hybrid_cli.py search "query" --mode vector

# Ingest all memory files + TOOLS.md/MEMORY.md/AGENTS.md
uv run python skills/hybrid-memory/scripts/hybrid_cli.py ingest-memory

# Ingest a directory
uv run python skills/hybrid-memory/scripts/hybrid_cli.py ingest --path memory/ --pattern "*.md"
uv run python skills/hybrid-memory/scripts/hybrid_cli.py ingest --path memory/ --incremental

# Store a single document
uv run python skills/hybrid-memory/scripts/hybrid_cli.py store --doc-id ID --text "content"
uv run python skills/hybrid-memory/scripts/hybrid_cli.py store --doc-id ID --file path/to/file.md

# Other
uv run python skills/hybrid-memory/scripts/hybrid_cli.py delete --doc-id ID
uv run python skills/hybrid-memory/scripts/hybrid_cli.py reindex
uv run python skills/hybrid-memory/scripts/hybrid_cli.py stats
```

## When to Use

- **Hybrid search**: exact keyword matches + semantic similarity. Best for credential lookups, config values, specific terms.
- **Tiered memory**: LLM-powered tree navigation. Best for broad semantic questions, "what happened last week".

Use hybrid search first (fast, precise), fall back to tiered memory if results are insufficient.

## Architecture

- **Chunker**: Markdown-aware splitting with heading context, configurable size/overlap
- **FTS5**: SQLite full-text search with BM25 ranking
- **Vector**: Ollama `nomic-embed-text` embeddings + cosine similarity
- **Merger**: Weighted combination (keyword=0.3, vector=0.7), normalization, deduplication

Gracefully degrades to keyword-only when Ollama is unavailable.

## Config

Edit `config.json` or pass CLI flags. DB stored in `data/hybrid_search.db`.
