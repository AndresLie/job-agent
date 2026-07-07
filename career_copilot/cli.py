from __future__ import annotations

import argparse
import json
from pathlib import Path

from .answer import answer_query
from .brief import generate_brief
from .documents import chunk_document, discover_documents, load_document
from .embeddings import build_embedder
from .evaluate import evaluate_retrieval
from .memory import MemoryStore
from .vector_store import JsonVectorStore, clear_storage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "raw"
DEFAULT_STORAGE = PROJECT_ROOT / "storage"
DEFAULT_INDEX = DEFAULT_STORAGE / "vector_store.json"
DEFAULT_MEMORY = DEFAULT_STORAGE / "memory.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="career-copilot")
    parser.add_argument("--embedder", choices=["hashing", "sentence-transformers"], default="hashing")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ingest = sub.add_parser("ingest", help="Index documents for retrieval")
    ingest.add_argument("--source", type=Path, default=DEFAULT_DATA)
    ingest.add_argument("--rebuild", action="store_true")

    ask = sub.add_parser("ask", help="Ask a cited question over indexed evidence")
    ask.add_argument("query")
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument("--no-llm", action="store_true")

    remember = sub.add_parser("remember", help="Store a durable preference or fact")
    remember.add_argument("--summary", required=True)
    remember.add_argument("--tags", default="")

    brief = sub.add_parser("brief", help="Generate a structured job fit brief")
    brief.add_argument("--job", type=Path, required=True)
    brief.add_argument("--top-k", type=int, default=8)
    brief.add_argument("--write-contract", action="store_true")

    evaluate = sub.add_parser("evaluate", help="Evaluate retrieval quality")
    evaluate.add_argument("--queries", type=Path, default=PROJECT_ROOT / "benchmarks" / "queries.jsonl")
    evaluate.add_argument("--k", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    embedder = build_embedder(args.embedder)
    store = JsonVectorStore(DEFAULT_INDEX)
    memory = MemoryStore(DEFAULT_MEMORY)

    if args.cmd == "ingest":
        if args.rebuild:
            clear_storage(DEFAULT_STORAGE)
            store = JsonVectorStore(DEFAULT_INDEX)
            memory = MemoryStore(DEFAULT_MEMORY)
        docs = discover_documents(args.source)
        chunks = []
        root = args.source if args.source.is_dir() else args.source.parent
        for doc in docs:
            chunks.extend(chunk_document(doc, load_document(doc), root=root))
        store.upsert_chunks(chunks, embedder)
        print(f"Indexed {len(chunks)} chunks from {len(docs)} documents into {DEFAULT_INDEX}.")
        return 0

    if args.cmd == "ask":
        result = answer_query(args.query, store, embedder, memory, top_k=args.top_k, use_llm=not args.no_llm)
        print(result["answer"])
        print("\nCitations")
        for index, citation in enumerate(result["citations"], start=1):
            print(f"- [{index}] {citation['source']} chunk {citation['chunk_index']} score {citation['score']}")
        return 0

    if args.cmd == "remember":
        tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
        added = memory.add(args.summary, tags=tags)
        print("Remembered." if added else "Already remembered.")
        return 0

    if args.cmd == "brief":
        payload = generate_brief(args.job, store, embedder, memory, top_k=args.top_k, write_contract=args.write_contract)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "evaluate":
        payload = evaluate_retrieval(args.queries, store, embedder, k=args.k)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    return 2
