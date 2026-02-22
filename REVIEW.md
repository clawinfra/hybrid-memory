# Hybrid Search Memory — Review Report

**Reviewer:** Subagent (reviewer-hybrid-memory)  
**Date:** 2026-02-22  
**Verdict:** ✅ PASS — solid implementation, production-ready

## Test Results

- **78 tests, all passing** (1.8s)
- **95% code coverage** (was 74% before review; added 27 tests)
- **Lint: clean** (fixed 5 ruff warnings)
- **Dogfooding: working** — 117 docs, 3748 chunks indexed and searchable

## Deviations from PLAN.md

| Deviation | Severity | Action |
|-----------|----------|--------|
| Tests in single `scripts/test_hybrid.py` instead of 7 files in `tests/` | Low | Acceptable — all test cases from plan are covered |
| CLI named `hybrid_cli.py` not `hybrid_memory_cli.py` | Low | Acceptable — shorter, consistent |
| Extra `ingest-memory` command not in plan | None | Good addition |

## Bugs Found & Fixed

1. **Lint: 5 warnings** — unused imports (`math`, `struct`, `Chunk`), unused variables (`results`, `p_ingest_mem`). All fixed.

## Security Review

- ✅ **SQL injection safe**: All queries use parameterized `?` placeholders
- ✅ **FTS5 query errors**: Caught via `try/except sqlite3.OperationalError`, returns `[]`
- ✅ **Path traversal**: `ingest` uses `glob.glob` on user-specified dir — standard CLI behavior, no web-facing risk
- ✅ **Ollama failure**: Graceful degradation to keyword-only. Tested and verified.

## Code Quality

- **Architecture**: Clean separation (chunker/fts/vector/merger/store/embedder). Matches plan exactly.
- **API signatures**: All match PLAN.md spec.
- **SQL schema**: Matches plan (WAL, FTS5, documents table).
- **Scoring formula**: Merger correctly normalizes, weights (0.3/0.7), deduplicates with "hybrid" label.
- **Chunker**: Correctly handles headings, overlap, force-splitting of long lines.
- **Embedder**: LRU cache with OrderedDict, MD5 keying, 30s timeout. Correct.
- **Embedding encoding**: float32 little-endian as specified (3072 bytes/chunk).

## Search Quality Observations

- `"SSH password peter"` → correctly finds TOOLS.md and memory entries with SSH info
- `"GPU server SSH"` → finds relevant content but vector results could rank TOOLS.md higher
- This is expected behavior — vector quality depends on nomic-embed-text model characteristics

## Coverage Details

| Module | Coverage |
|--------|----------|
| chunker.py | 100% |
| embedder.py | 100% |
| fts.py | 100% |
| vector.py | 100% |
| merger.py | 100% |
| store.py | 90% |
| hybrid_cli.py | 79% |
| **TOTAL** | **95%** |

Uncovered lines in `hybrid_cli.py` are `cmd_ingest_memory` (hard to test without real workspace) and `main()` argv parsing edge cases. Core logic is 100% covered.

## Tests Added by Reviewer

- `TestFTSSnippet` (3 tests) — snippet method coverage
- `TestStoreClose` (1 test) — close/cleanup
- `TestEdgeCases` (10 tests) — SQL injection, empty queries, long queries, binary content, unicode, metadata
- `TestCLIFunctions` (12 tests) — direct CLI function testing for coverage
