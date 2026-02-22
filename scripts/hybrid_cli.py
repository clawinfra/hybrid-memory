#!/usr/bin/env python3
"""CLI entry point for hybrid search memory."""
import argparse
import glob
import json
import logging
import os
import sys

# Add parent to path so we can import as package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.store import HybridStore


def load_config(args) -> dict:
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json")
    config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)

    # CLI overrides
    if hasattr(args, "ollama_url") and args.ollama_url:
        config["ollama_url"] = args.ollama_url
    if hasattr(args, "model") and args.model:
        config["model"] = args.model
    if hasattr(args, "chunk_size") and args.chunk_size:
        config["chunk_size"] = args.chunk_size
    if hasattr(args, "overlap") and args.overlap:
        config["chunk_overlap"] = args.overlap
    if hasattr(args, "keyword_weight") and args.keyword_weight is not None:
        config["keyword_weight"] = args.keyword_weight
    if hasattr(args, "vector_weight") and args.vector_weight is not None:
        config["vector_weight"] = args.vector_weight
    return config


def get_db_path(args) -> str:
    if args.db:
        return args.db
    skill_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(skill_dir, "data", "hybrid_search.db")


def cmd_store(args):
    config = load_config(args)
    store = HybridStore(db_path=get_db_path(args), config=config)
    try:
        if args.file:
            with open(args.file) as f:
                text = f.read()
        else:
            text = args.text
        metadata = {}
        if args.file:
            metadata["source_path"] = args.file
            metadata["file_mtime"] = str(os.path.getmtime(args.file))
        n = store.store(args.doc_id, text, metadata=metadata)
        print(json.dumps({"doc_id": args.doc_id, "chunks": n}))
    finally:
        store.close()


def cmd_search(args):
    config = load_config(args)
    store = HybridStore(db_path=get_db_path(args), config=config)
    try:
        results = store.search(args.query, limit=args.limit, mode=args.mode)
        print(json.dumps(results, indent=2))
    finally:
        store.close()


def cmd_ingest(args):
    config = load_config(args)
    db_path = get_db_path(args)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    store = HybridStore(db_path=db_path, config=config)
    try:
        pattern = os.path.join(args.path, "**", args.pattern)
        files = glob.glob(pattern, recursive=True)
        ingested = 0
        total_chunks = 0
        skipped = 0
        errors = []

        for fpath in sorted(files):
            doc_id = os.path.relpath(fpath)
            mtime = os.path.getmtime(fpath)

            if args.incremental:
                row = store.conn.execute(
                    "SELECT file_mtime FROM documents WHERE doc_id = ?", (doc_id,)
                ).fetchone()
                if row and row[0] == mtime:
                    skipped += 1
                    continue

            try:
                with open(fpath) as f:
                    text = f.read()
                n = store.store(doc_id, text, metadata={"source_path": fpath, "file_mtime": str(mtime)})
                total_chunks += n
                ingested += 1
            except Exception as e:
                errors.append({"file": fpath, "error": str(e)})

        print(json.dumps({"files": ingested, "chunks": total_chunks, "skipped": skipped, "errors": errors}))
    finally:
        store.close()


def cmd_ingest_memory(args):
    """Convenience: ingest memory/ dir + key workspace docs."""
    config = load_config(args)
    db_path = get_db_path(args)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    store = HybridStore(db_path=db_path, config=config)
    workspace = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    try:
        total_files = 0
        total_chunks = 0

        # Ingest memory/*.md
        mem_dir = os.path.join(workspace, "memory")
        if os.path.isdir(mem_dir):
            for fpath in sorted(glob.glob(os.path.join(mem_dir, "**", "*.md"), recursive=True)):
                doc_id = os.path.relpath(fpath, workspace)
                try:
                    with open(fpath) as f:
                        text = f.read()
                    n = store.store(doc_id, text, metadata={"source_path": fpath, "file_mtime": str(os.path.getmtime(fpath))})
                    total_chunks += n
                    total_files += 1
                except Exception as e:
                    print(f"Warning: {fpath}: {e}", file=sys.stderr)

        # Ingest key workspace docs
        for fname in ["TOOLS.md", "MEMORY.md", "AGENTS.md"]:
            fpath = os.path.join(workspace, fname)
            if os.path.exists(fpath):
                try:
                    with open(fpath) as f:
                        text = f.read()
                    n = store.store(fname, text, metadata={"source_path": fpath, "file_mtime": str(os.path.getmtime(fpath))})
                    total_chunks += n
                    total_files += 1
                except Exception as e:
                    print(f"Warning: {fname}: {e}", file=sys.stderr)

        print(json.dumps({"files": total_files, "chunks": total_chunks}))
    finally:
        store.close()


def cmd_delete(args):
    config = load_config(args)
    store = HybridStore(db_path=get_db_path(args), config=config)
    try:
        n = store.delete(args.doc_id)
        print(json.dumps({"doc_id": args.doc_id, "chunks_deleted": n}))
    finally:
        store.close()


def cmd_reindex(args):
    config = load_config(args)
    store = HybridStore(db_path=get_db_path(args), config=config)
    try:
        store.reindex()
        count = store.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        print(json.dumps({"status": "ok", "chunks": count}))
    finally:
        store.close()


def cmd_stats(args):
    config = load_config(args)
    store = HybridStore(db_path=get_db_path(args), config=config)
    try:
        s = store.stats()
        print(json.dumps(s, indent=2))
    finally:
        store.close()


def main():
    parser = argparse.ArgumentParser(prog="hybrid_cli", description="Hybrid search memory")
    parser.add_argument("--db", default=None, help="SQLite database path")
    parser.add_argument("--ollama-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--keyword-weight", type=float, default=None)
    parser.add_argument("--vector-weight", type=float, default=None)
    parser.add_argument("--quiet", action="store_true")

    sub = parser.add_subparsers(dest="command")

    p_store = sub.add_parser("store")
    p_store.add_argument("--doc-id", required=True)
    p_store.add_argument("--text", default=None)
    p_store.add_argument("--file", default=None)

    p_search = sub.add_parser("search")
    p_search.add_argument("query", nargs="?", default=None)
    p_search.add_argument("--query", dest="query_flag", default=None)
    p_search.add_argument("--limit", type=int, default=10)
    p_search.add_argument("--mode", default="hybrid", choices=["hybrid", "keyword", "vector"])

    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("--path", required=True)
    p_ingest.add_argument("--pattern", default="*.md")
    p_ingest.add_argument("--incremental", action="store_true")

    sub.add_parser("ingest-memory")

    p_delete = sub.add_parser("delete")
    p_delete.add_argument("--doc-id", required=True)

    sub.add_parser("reindex")
    sub.add_parser("stats")

    args = parser.parse_args()

    if args.quiet:
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Resolve search query from positional or --query flag
    if args.command == "search":
        if args.query is None and args.query_flag:
            args.query = args.query_flag
        if not args.query:
            print("Error: query required", file=sys.stderr)
            sys.exit(1)

    commands = {
        "store": cmd_store,
        "search": cmd_search,
        "ingest": cmd_ingest,
        "ingest-memory": cmd_ingest_memory,
        "delete": cmd_delete,
        "reindex": cmd_reindex,
        "stats": cmd_stats,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
