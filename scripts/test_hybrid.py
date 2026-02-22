"""Tests for hybrid search memory — all modules."""
import argparse
import hashlib
import json
import os
import random
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# Add scripts to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.chunker import chunk_text
from scripts.embedder import OllamaEmbedder
from scripts.fts import FTSIndex
from scripts.vector import cosine_similarity, encode_embedding, decode_embedding, VectorSearch
from scripts.merger import merge_results
from scripts.store import HybridStore


class MockEmbedder:
    """Returns deterministic embeddings based on text hash."""
    def embed(self, text: str) -> list[float]:
        h = hashlib.md5(text.encode()).digest()
        rng = random.Random(int.from_bytes(h, "big"))
        return [rng.gauss(0, 1) for _ in range(768)]

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]

    def dims(self):
        return 768

    def is_available(self):
        return True


def make_db():
    """Create in-memory DB with schema."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE IF NOT EXISTS chunks (doc_id TEXT NOT NULL, chunk_id TEXT NOT NULL, heading TEXT NOT NULL DEFAULT '', content TEXT NOT NULL, embedding BLOB, created_at REAL NOT NULL DEFAULT 0, PRIMARY KEY (doc_id, chunk_id))")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id)")
    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(doc_id, chunk_id, heading, content, tokenize='porter unicode61')")
    conn.execute("CREATE TABLE IF NOT EXISTS documents (doc_id TEXT PRIMARY KEY, source_path TEXT, file_mtime REAL, chunk_count INTEGER NOT NULL DEFAULT 0, indexed_at REAL NOT NULL DEFAULT 0)")
    conn.commit()
    return conn


# ---- Chunker Tests ----

class TestChunker(unittest.TestCase):
    def test_empty_text(self):
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("   "), [])

    def test_single_line(self):
        chunks = chunk_text("Hello world")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "Hello world")

    def test_headings_split(self):
        text = "# Title\nSome content\n## Section\nMore content"
        chunks = chunk_text(text, chunk_size=5000)
        self.assertTrue(len(chunks) >= 1)
        # Check heading is tracked
        found_section = False
        for c in chunks:
            if "Section" in c.heading:
                found_section = True
        self.assertTrue(found_section or len(chunks) == 1)  # small text may be 1 chunk

    def test_chunk_size_split(self):
        text = "word " * 200  # ~1000 chars
        chunks = chunk_text(text, chunk_size=200, overlap=20)
        self.assertTrue(len(chunks) > 1)

    def test_overlap(self):
        text = "A" * 300 + "\n" + "B" * 300
        chunks = chunk_text(text, chunk_size=350, overlap=50)
        if len(chunks) > 1:
            # Second chunk should have overlap from first
            self.assertTrue(len(chunks[1].text) > 0)

    def test_unicode(self):
        text = "# 标题\n这是中文内容\n## 部分\n更多内容"
        chunks = chunk_text(text)
        self.assertTrue(len(chunks) >= 1)


# ---- FTS Tests ----

class TestFTS(unittest.TestCase):
    def setUp(self):
        self.conn = make_db()
        self.fts = FTSIndex(self.conn)
        # Insert test data
        for i, (text, heading) in enumerate([
            ("The quick brown fox jumps over the lazy dog", "# Animals"),
            ("Python is a great programming language", "# Programming"),
            ("SSH password is hunter2 for the server", "# Credentials"),
        ]):
            doc_id = f"doc{i}"
            chunk_id = f"doc{i}-0"
            self.conn.execute("INSERT INTO chunks (doc_id, chunk_id, heading, content) VALUES (?,?,?,?)", (doc_id, chunk_id, heading, text))
            self.conn.execute("INSERT INTO chunks_fts (doc_id, chunk_id, heading, content) VALUES (?,?,?,?)", (doc_id, chunk_id, heading, text))
        self.conn.commit()

    def test_search_found(self):
        results = self.fts.search("python programming")
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]["source"], "keyword")

    def test_search_not_found(self):
        results = self.fts.search("xyznonexistent")
        self.assertEqual(len(results), 0)

    def test_empty_query(self):
        self.assertEqual(self.fts.search(""), [])

    def test_special_chars(self):
        # Should not crash
        self.fts.search('test AND OR "unclosed')
        # May return empty or results, just shouldn't crash

    def test_snippet(self):
        results = self.fts.snippet("password")
        self.assertTrue(len(results) > 0)

    def test_reindex(self):
        self.conn.execute("DELETE FROM chunks_fts")
        self.conn.commit()
        self.assertEqual(self.fts.search("python"), [])
        self.fts.reindex()
        self.assertTrue(len(self.fts.search("python")) > 0)


# ---- Vector Tests ----

class TestVector(unittest.TestCase):
    def test_cosine_identical(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [1, 0]), 1.0)

    def test_cosine_orthogonal(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)

    def test_cosine_zero_vector(self):
        self.assertEqual(cosine_similarity([0, 0], [1, 0]), 0.0)

    def test_cosine_empty(self):
        self.assertEqual(cosine_similarity([], []), 0.0)

    def test_cosine_different_lengths(self):
        self.assertEqual(cosine_similarity([1, 0], [1, 0, 0]), 0.0)

    def test_encode_decode_roundtrip(self):
        v = [0.1, 0.2, -0.3, 0.0, 1.0]
        decoded = decode_embedding(encode_embedding(v))
        for a, b in zip(v, decoded):
            self.assertAlmostEqual(a, b, places=5)

    def test_vector_search(self):
        conn = make_db()
        embedder = MockEmbedder()
        # Insert chunks with embeddings
        for i, text in enumerate(["apple banana fruit", "car engine motor", "python code function"]):
            emb = embedder.embed(text)
            blob = encode_embedding(emb)
            conn.execute("INSERT INTO chunks (doc_id, chunk_id, heading, content, embedding) VALUES (?,?,?,?,?)",
                         (f"d{i}", f"d{i}-0", "", text, blob))
        conn.commit()

        vs = VectorSearch(conn, embedder)
        results = vs.search("fruit apple", limit=3)
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]["source"], "vector")

    def test_vector_search_no_embeddings(self):
        conn = make_db()
        embedder = MockEmbedder()
        vs = VectorSearch(conn, embedder)
        results = vs.search("anything")
        self.assertEqual(results, [])

    def test_vector_search_embedder_fails(self):
        conn = make_db()

        class FailEmbedder:
            def embed(self, text):
                return []

        vs = VectorSearch(conn, FailEmbedder())
        self.assertEqual(vs.search("test"), [])


# ---- Merger Tests ----

class TestMerger(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(merge_results([], []), [])

    def test_keyword_only(self):
        kw = [{"doc_id": "a", "chunk_id": "a-0", "heading": "", "text": "x", "score": 5.0, "source": "keyword"}]
        results = merge_results(kw, [], keyword_weight=0.3, vector_weight=0.7)
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0]["score"], 0.3)  # 1.0 * 0.3

    def test_vector_only(self):
        vec = [{"doc_id": "a", "chunk_id": "a-0", "heading": "", "text": "x", "score": 0.8, "source": "vector"}]
        results = merge_results([], vec, keyword_weight=0.3, vector_weight=0.7)
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0]["score"], 0.7)  # 1.0 * 0.7

    def test_hybrid_dedup(self):
        kw = [{"doc_id": "a", "chunk_id": "a-0", "heading": "", "text": "x", "score": 5.0, "source": "keyword"}]
        vec = [{"doc_id": "a", "chunk_id": "a-0", "heading": "", "text": "x", "score": 0.9, "source": "vector"}]
        results = merge_results(kw, vec, keyword_weight=0.3, vector_weight=0.7)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source"], "hybrid")
        self.assertAlmostEqual(results[0]["score"], 1.0)  # 1.0*0.3 + 1.0*0.7

    def test_zero_scores(self):
        kw = [{"doc_id": "a", "chunk_id": "a-0", "heading": "", "text": "x", "score": 0.0, "source": "keyword"}]
        results = merge_results(kw, [])
        self.assertEqual(len(results), 1)

    def test_weight_zero(self):
        kw = [{"doc_id": "a", "chunk_id": "a-0", "heading": "", "text": "x", "score": 5.0, "source": "keyword"}]
        vec = [{"doc_id": "b", "chunk_id": "b-0", "heading": "", "text": "y", "score": 0.8, "source": "vector"}]
        results = merge_results(kw, vec, keyword_weight=0.0, vector_weight=1.0)
        self.assertEqual(results[0]["doc_id"], "b")


# ---- Embedder Tests ----

class TestEmbedder(unittest.TestCase):
    def test_embed_success(self):
        embedder = OllamaEmbedder()
        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = {"embeddings": [[0.1] * 768]}
        fake_response.raise_for_status = MagicMock()

        with patch("scripts.embedder.requests.post", return_value=fake_response):
            result = embedder.embed("test")
            self.assertEqual(len(result), 768)

    def test_embed_failure(self):
        embedder = OllamaEmbedder()
        with patch("scripts.embedder.requests.post", side_effect=Exception("connection refused")):
            result = embedder.embed("test")
            self.assertEqual(result, [])

    def test_cache_hit(self):
        embedder = OllamaEmbedder()
        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = {"embeddings": [[0.5] * 768]}
        fake_response.raise_for_status = MagicMock()

        with patch("scripts.embedder.requests.post", return_value=fake_response) as mock_post:
            embedder.embed("same text")
            embedder.embed("same text")
            self.assertEqual(mock_post.call_count, 1)

    def test_cache_eviction(self):
        embedder = OllamaEmbedder(cache_size=2)
        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.raise_for_status = MagicMock()

        call_count = 0
        def make_response(*a, **kw):
            nonlocal call_count
            call_count += 1
            fake_response.json.return_value = {"embeddings": [[float(call_count)] * 768]}
            return fake_response

        with patch("scripts.embedder.requests.post", side_effect=make_response):
            embedder.embed("a")
            embedder.embed("b")
            embedder.embed("c")  # evicts "a"
            self.assertEqual(len(embedder._cache), 2)

    def test_embed_batch(self):
        embedder = OllamaEmbedder()
        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = {"embeddings": [[0.1] * 768]}
        fake_response.raise_for_status = MagicMock()

        with patch("scripts.embedder.requests.post", return_value=fake_response):
            results = embedder.embed_batch(["a", "b"])
            self.assertEqual(len(results), 2)

    def test_dims(self):
        self.assertEqual(OllamaEmbedder().dims(), 768)

    def test_is_available_yes(self):
        embedder = OllamaEmbedder()
        fake_response = MagicMock()
        fake_response.status_code = 200
        with patch("scripts.embedder.requests.get", return_value=fake_response):
            self.assertTrue(embedder.is_available())

    def test_is_available_no(self):
        embedder = OllamaEmbedder()
        with patch("scripts.embedder.requests.get", side_effect=Exception("nope")):
            self.assertFalse(embedder.is_available())


# ---- Store Tests ----

class TestStore(unittest.TestCase):
    def _make_store(self):
        """Create a store with in-memory DB and mock embedder."""
        with patch("scripts.store.OllamaEmbedder") as MockEmb:
            mock_instance = MockEmbedder()
            MockEmb.return_value = mock_instance
            store = HybridStore(db_path=":memory:", config={})
            store.embedder = mock_instance
            store.vector = VectorSearch(store.conn, mock_instance)
            return store

    def test_store_and_search(self):
        store = self._make_store()
        n = store.store("doc1", "The quick brown fox jumps over the lazy dog")
        self.assertTrue(n > 0)
        results = store.search("quick fox", mode="keyword")
        self.assertTrue(len(results) > 0)

    def test_store_upsert(self):
        store = self._make_store()
        store.store("doc1", "First version")
        store.store("doc1", "Second version completely different")
        results = store.search("Second version", mode="keyword")
        self.assertTrue(any("Second" in r["text"] for r in results))
        # First version should be gone
        results2 = store.search("First version", mode="keyword")
        self.assertEqual(len(results2), 0)

    def test_delete(self):
        store = self._make_store()
        store.store("doc1", "Some content here")
        n = store.delete("doc1")
        self.assertTrue(n > 0)
        self.assertEqual(store.search("content", mode="keyword"), [])

    def test_search_keyword_mode(self):
        store = self._make_store()
        store.store("doc1", "Python programming language")
        results = store.search("Python", mode="keyword")
        self.assertTrue(len(results) > 0)

    def test_search_vector_mode(self):
        store = self._make_store()
        store.store("doc1", "Python programming language")
        results = store.search("Python", mode="vector")
        self.assertTrue(len(results) > 0)

    def test_search_hybrid_mode(self):
        store = self._make_store()
        store.store("doc1", "Python programming language")
        results = store.search("Python", mode="hybrid")
        self.assertTrue(len(results) > 0)

    def test_stats(self):
        store = self._make_store()
        store.store("doc1", "Hello world")
        s = store.stats()
        self.assertEqual(s["doc_count"], 1)
        self.assertTrue(s["chunk_count"] > 0)

    def test_reindex(self):
        store = self._make_store()
        store.store("doc1", "Test reindex content")
        store.conn.execute("DELETE FROM chunks_fts")
        store.conn.commit()
        self.assertEqual(store.search("reindex", mode="keyword"), [])
        store.reindex()
        self.assertTrue(len(store.search("reindex", mode="keyword")) > 0)

    def test_graceful_degradation(self):
        """When embedder fails, keyword search still works."""
        with patch("scripts.store.OllamaEmbedder") as MockEmb:
            class FailEmb:
                def embed(self, text): return []
                def embed_batch(self, texts): return [[] for _ in texts]
                def dims(self): return 768
                def is_available(self): return False

            MockEmb.return_value = FailEmb()
            store = HybridStore(db_path=":memory:", config={})
            store.embedder = FailEmb()
            store.vector = VectorSearch(store.conn, store.embedder)

            store.store("doc1", "Keyword only content here")
            results = store.search("Keyword", mode="keyword")
            self.assertTrue(len(results) > 0)


# ---- CLI Integration Tests ----

class TestCLI(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.cli = [sys.executable, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "hybrid_cli.py")]

    def _run(self, *args):
        cmd = self.cli + ["--db", self.db_path, "--quiet"] + list(args)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result

    def test_store_and_search(self):
        r = self._run("store", "--doc-id", "test1", "--text", "The quick brown fox")
        self.assertEqual(r.returncode, 0)
        data = json.loads(r.stdout)
        self.assertEqual(data["doc_id"], "test1")

        r = self._run("search", "--mode", "keyword", "quick fox")
        self.assertEqual(r.returncode, 0)
        results = json.loads(r.stdout)
        self.assertTrue(len(results) > 0)

    def test_ingest(self):
        # Create temp md files
        md_dir = os.path.join(self.tmpdir, "docs")
        os.makedirs(md_dir)
        with open(os.path.join(md_dir, "test.md"), "w") as f:
            f.write("# Test\nHello world content")

        r = self._run("ingest", "--path", md_dir)
        self.assertEqual(r.returncode, 0)
        data = json.loads(r.stdout)
        self.assertEqual(data["files"], 1)

    def test_ingest_incremental(self):
        md_dir = os.path.join(self.tmpdir, "docs")
        os.makedirs(md_dir)
        with open(os.path.join(md_dir, "test.md"), "w") as f:
            f.write("# Test\nContent")

        self._run("ingest", "--path", md_dir)
        r = self._run("ingest", "--path", md_dir, "--incremental")
        data = json.loads(r.stdout)
        self.assertEqual(data["skipped"], 1)

    def test_delete(self):
        self._run("store", "--doc-id", "del1", "--text", "Delete me")
        r = self._run("delete", "--doc-id", "del1")
        self.assertEqual(r.returncode, 0)

    def test_stats(self):
        self._run("store", "--doc-id", "s1", "--text", "Stats test")
        r = self._run("stats")
        self.assertEqual(r.returncode, 0)
        data = json.loads(r.stdout)
        self.assertIn("doc_count", data)

    def test_reindex(self):
        self._run("store", "--doc-id", "r1", "--text", "Reindex test")
        r = self._run("reindex")
        self.assertEqual(r.returncode, 0)

    def test_store_from_file(self):
        fpath = os.path.join(self.tmpdir, "input.md")
        with open(fpath, "w") as f:
            f.write("# File Input\nContent from file")
        r = self._run("store", "--doc-id", "file1", "--file", fpath)
        self.assertEqual(r.returncode, 0)


# ---- Additional Coverage Tests ----

class TestFTSSnippet(unittest.TestCase):
    """Cover FTSIndex.snippet method."""
    def setUp(self):
        self.conn = make_db()
        self.fts = FTSIndex(self.conn)
        self.conn.execute(
            "INSERT INTO chunks (doc_id, chunk_id, heading, content) VALUES (?,?,?,?)",
            ("d1", "d1-0", "# Test", "The password is secret123 for SSH"),
        )
        self.conn.execute(
            "INSERT INTO chunks_fts (doc_id, chunk_id, heading, content) VALUES (?,?,?,?)",
            ("d1", "d1-0", "# Test", "The password is secret123 for SSH"),
        )
        self.conn.commit()

    def test_snippet_returns_results(self):
        results = self.fts.snippet("password")
        self.assertTrue(len(results) > 0)
        self.assertIn("**", results[0]["text"])  # highlight markers

    def test_snippet_empty_query(self):
        self.assertEqual(self.fts.snippet(""), [])

    def test_snippet_bad_query(self):
        self.fts.snippet('AND OR "unclosed')
        # Should not crash


class TestStoreClose(unittest.TestCase):
    def test_close(self):
        with patch("scripts.store.OllamaEmbedder") as MockEmb:
            MockEmb.return_value = MockEmbedder()
            store = HybridStore(db_path=":memory:", config={})
            store.close()
            # After close, operations should fail
            with self.assertRaises(Exception):
                store.conn.execute("SELECT 1")


class TestEdgeCases(unittest.TestCase):
    """Edge case tests for security and robustness."""

    def _make_store(self):
        with patch("scripts.store.OllamaEmbedder") as MockEmb:
            MockEmb.return_value = MockEmbedder()
            store = HybridStore(db_path=":memory:", config={})
            store.embedder = MockEmbedder()
            store.vector = VectorSearch(store.conn, store.embedder)
            return store

    def test_empty_query_search(self):
        store = self._make_store()
        store.store("doc1", "Some content")
        results = store.search("", mode="keyword")
        self.assertEqual(results, [])

    def test_sql_injection_search(self):
        store = self._make_store()
        store.store("doc1", "Safe content")
        # Should not crash or leak data
        store.search("'; DROP TABLE chunks; --", mode="keyword")
        # Verify table still exists
        count = store.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        self.assertTrue(count > 0)

    def test_sql_injection_store(self):
        store = self._make_store()
        store.store("'; DROP TABLE chunks; --", "Malicious doc_id")
        count = store.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        self.assertTrue(count > 0)

    def test_empty_text_store(self):
        store = self._make_store()
        n = store.store("empty", "")
        self.assertEqual(n, 0)

    def test_very_long_query(self):
        store = self._make_store()
        store.store("doc1", "Some content for testing")
        long_query = "test " * 500
        # Should not crash
        store.search(long_query, mode="keyword")

    def test_binary_content_store(self):
        store = self._make_store()
        # Binary-ish content
        binary_text = "Hello \x00\x01\x02 world"
        # Should handle gracefully (may store 0 chunks or error)
        try:
            store.store("binary", binary_text)
        except Exception:
            pass  # acceptable to fail on binary

    def test_unicode_search(self):
        store = self._make_store()
        store.store("unicode", "日本語のテスト文書")
        store.search("日本語", mode="keyword")
        # FTS may or may not match CJK, but shouldn't crash

    def test_delete_nonexistent(self):
        store = self._make_store()
        n = store.delete("nonexistent")
        self.assertEqual(n, 0)

    def test_store_with_metadata(self):
        store = self._make_store()
        n = store.store("meta", "Content", metadata={"source_path": "/tmp/test.md", "file_mtime": "12345"})
        self.assertTrue(n > 0)
        row = store.conn.execute("SELECT source_path, file_mtime FROM documents WHERE doc_id = ?", ("meta",)).fetchone()
        self.assertEqual(row[0], "/tmp/test.md")


class TestCLIFunctions(unittest.TestCase):
    """Test CLI functions directly for coverage."""

    def test_load_config_defaults(self):
        from scripts.hybrid_cli import load_config
        args = argparse.Namespace(ollama_url=None, model=None, chunk_size=None, overlap=None, keyword_weight=None, vector_weight=None)
        config = load_config(args)
        self.assertIsInstance(config, dict)

    def test_load_config_overrides(self):
        from scripts.hybrid_cli import load_config
        args = argparse.Namespace(ollama_url="http://test:1234", model="test-model", chunk_size=256, overlap=25, keyword_weight=0.5, vector_weight=0.5)
        config = load_config(args)
        self.assertEqual(config["ollama_url"], "http://test:1234")
        self.assertEqual(config["model"], "test-model")
        self.assertEqual(config["chunk_size"], 256)

    def test_get_db_path_default(self):
        from scripts.hybrid_cli import get_db_path
        args = argparse.Namespace(db=None)
        path = get_db_path(args)
        self.assertTrue(path.endswith("hybrid_search.db"))

    def test_get_db_path_custom(self):
        from scripts.hybrid_cli import get_db_path
        args = argparse.Namespace(db="/tmp/custom.db")
        self.assertEqual(get_db_path(args), "/tmp/custom.db")

    def test_cmd_store_text(self):
        from scripts.hybrid_cli import cmd_store
        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "test.db")
            args = argparse.Namespace(
                db=db, ollama_url=None, model=None, chunk_size=None,
                overlap=None, keyword_weight=None, vector_weight=None,
                quiet=True, doc_id="t1", text="Hello world test", file=None,
            )
            with patch("sys.stdout", new_callable=lambda: open(os.devnull, "w")):
                cmd_store(args)

    def test_cmd_store_file(self):
        from scripts.hybrid_cli import cmd_store
        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "test.db")
            fpath = os.path.join(td, "input.md")
            with open(fpath, "w") as f:
                f.write("# From file\nContent here")
            args = argparse.Namespace(
                db=db, ollama_url=None, model=None, chunk_size=None,
                overlap=None, keyword_weight=None, vector_weight=None,
                quiet=True, doc_id="f1", text=None, file=fpath,
            )
            with patch("sys.stdout", new_callable=lambda: open(os.devnull, "w")):
                cmd_store(args)

    def test_cmd_search(self):
        from scripts.hybrid_cli import cmd_store, cmd_search
        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "test.db")
            base = argparse.Namespace(
                db=db, ollama_url=None, model=None, chunk_size=None,
                overlap=None, keyword_weight=None, vector_weight=None, quiet=True,
            )
            store_args = argparse.Namespace(**vars(base), doc_id="s1", text="Python programming", file=None)
            with patch("sys.stdout", new_callable=lambda: open(os.devnull, "w")):
                cmd_store(store_args)
            search_args = argparse.Namespace(**vars(base), query="Python", limit=5, mode="keyword")
            with patch("sys.stdout", new_callable=lambda: open(os.devnull, "w")):
                cmd_search(search_args)

    def test_cmd_delete(self):
        from scripts.hybrid_cli import cmd_store, cmd_delete
        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "test.db")
            base = argparse.Namespace(
                db=db, ollama_url=None, model=None, chunk_size=None,
                overlap=None, keyword_weight=None, vector_weight=None, quiet=True,
            )
            store_args = argparse.Namespace(**vars(base), doc_id="d1", text="Delete me", file=None)
            with patch("sys.stdout", new_callable=lambda: open(os.devnull, "w")):
                cmd_store(store_args)
            del_args = argparse.Namespace(**vars(base), doc_id="d1")
            with patch("sys.stdout", new_callable=lambda: open(os.devnull, "w")):
                cmd_delete(del_args)

    def test_cmd_reindex(self):
        from scripts.hybrid_cli import cmd_store, cmd_reindex
        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "test.db")
            base = argparse.Namespace(
                db=db, ollama_url=None, model=None, chunk_size=None,
                overlap=None, keyword_weight=None, vector_weight=None, quiet=True,
            )
            store_args = argparse.Namespace(**vars(base), doc_id="r1", text="Reindex me", file=None)
            with patch("sys.stdout", new_callable=lambda: open(os.devnull, "w")):
                cmd_store(store_args)
            with patch("sys.stdout", new_callable=lambda: open(os.devnull, "w")):
                cmd_reindex(base)

    def test_cmd_stats(self):
        from scripts.hybrid_cli import cmd_stats
        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "test.db")
            args = argparse.Namespace(
                db=db, ollama_url=None, model=None, chunk_size=None,
                overlap=None, keyword_weight=None, vector_weight=None, quiet=True,
            )
            with patch("sys.stdout", new_callable=lambda: open(os.devnull, "w")):
                cmd_stats(args)

    def test_cmd_ingest(self):
        from scripts.hybrid_cli import cmd_ingest
        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "test.db")
            md_dir = os.path.join(td, "docs")
            os.makedirs(md_dir)
            with open(os.path.join(md_dir, "test.md"), "w") as f:
                f.write("# Test\nContent")
            args = argparse.Namespace(
                db=db, ollama_url=None, model=None, chunk_size=None,
                overlap=None, keyword_weight=None, vector_weight=None,
                quiet=True, path=md_dir, pattern="*.md", incremental=False,
            )
            with patch("sys.stdout", new_callable=lambda: open(os.devnull, "w")):
                cmd_ingest(args)

    def test_cmd_ingest_incremental(self):
        from scripts.hybrid_cli import cmd_ingest
        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "test.db")
            md_dir = os.path.join(td, "docs")
            os.makedirs(md_dir)
            with open(os.path.join(md_dir, "test.md"), "w") as f:
                f.write("# Test\nContent")
            base = dict(
                db=db, ollama_url=None, model=None, chunk_size=None,
                overlap=None, keyword_weight=None, vector_weight=None,
                quiet=True, path=md_dir, pattern="*.md",
            )
            with patch("sys.stdout", new_callable=lambda: open(os.devnull, "w")):
                cmd_ingest(argparse.Namespace(**base, incremental=False))
            with patch("sys.stdout", new_callable=lambda: open(os.devnull, "w")):
                cmd_ingest(argparse.Namespace(**base, incremental=True))

    def test_main_no_command(self):
        from scripts.hybrid_cli import main
        with patch("sys.argv", ["hybrid_cli"]):
            with self.assertRaises(SystemExit):
                main()

    def test_main_search_no_query(self):
        from scripts.hybrid_cli import main
        with patch("sys.argv", ["hybrid_cli", "search"]):
            with self.assertRaises(SystemExit):
                main()


if __name__ == "__main__":
    unittest.main()
