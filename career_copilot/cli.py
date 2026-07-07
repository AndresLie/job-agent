from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .answer import answer_query
from .brief import generate_brief
from .config import load_env_file
from .documents import chunk_document, discover_documents, load_document
from .embeddings import build_embedder
from .evaluate import evaluate_retrieval
from .job_input import resolve_job_input
from .memory import MemoryStore
from .vector_store import JsonVectorStore, clear_storage
from .web_research import research_company


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "raw"
DEFAULT_STORAGE = PROJECT_ROOT / "storage"
DEFAULT_INDEX = DEFAULT_STORAGE / "vector_store.json"
DEFAULT_MEMORY = DEFAULT_STORAGE / "memory.json"
DEFAULT_JOBS = DEFAULT_DATA / "jobs"
DEFAULT_COMPANY_RESEARCH = DEFAULT_DATA / "company_research"


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

    recall = sub.add_parser("recall", help="Search durable memory without querying RAG documents")
    recall.add_argument("query")
    recall.add_argument("--k", type=int, default=5)
    recall.add_argument("--json", action="store_true", help="Print matching memories as JSON")

    brief = sub.add_parser("brief", help="Generate a structured job fit brief")
    brief.add_argument("--job", type=Path, default=None, help="Backward-compatible alias for --job-file")
    brief.add_argument("--job-file", type=Path, default=None)
    brief.add_argument("--job-text", default=None)
    brief.add_argument("--job-url", default=None)
    brief.add_argument("--stdin", action="store_true", help="Read job description text from stdin")
    brief.add_argument("--company", default=None)
    brief.add_argument("--research-company", action="store_true")
    brief.add_argument("--no-cache", action="store_true")
    brief.add_argument("--top-k", type=int, default=8)
    brief.add_argument("--write-contract", action="store_true")

    research = sub.add_parser("research-company", help="Fetch company/job context with Exa")
    research.add_argument("--company", required=True)
    research.add_argument("--role", default="AI Engineer")
    research.add_argument("--num-results", type=int, default=5)
    research.add_argument("--no-cache", action="store_true")

    evaluate = sub.add_parser("evaluate", help="Evaluate retrieval quality")
    evaluate.add_argument("--queries", type=Path, default=PROJECT_ROOT / "benchmarks" / "queries.jsonl")
    evaluate.add_argument("--k", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_env_file(PROJECT_ROOT / ".env")
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
            print(
                f"- [{index}] {citation['category']}/{citation['source']} "
                f"chunk {citation['chunk_index']} score {citation['score']}"
            )
        return 0

    if args.cmd == "remember":
        tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
        added = memory.add(args.summary, tags=tags)
        print("Remembered." if added else "Already remembered.")
        return 0

    if args.cmd == "recall":
        hits = memory.retrieve(args.query, k=args.k)
        if args.json:
            print(json.dumps([item.__dict__ for item in hits], ensure_ascii=False, indent=2))
            return 0
        if not hits:
            print("No matching memories found.")
            return 0
        for index, item in enumerate(hits, start=1):
            tag_text = f" tags={','.join(item.tags)}" if item.tags else ""
            print(f"[{index}] {item.summary}{tag_text}")
        return 0

    if args.cmd == "brief":
        job_file = args.job_file or args.job
        job = resolve_job_input(
            job_file=job_file,
            job_text=args.job_text,
            job_url=args.job_url,
            use_stdin=args.stdin,
            company=args.company,
            cache_dir=DEFAULT_JOBS,
            cache=not args.no_cache,
        )
        web_sources = []
        if args.research_company:
            company = job.company or args.company
            if not company:
                raise SystemExit("--research-company requires --company when it cannot be inferred from the job URL.")
            try:
                web_sources = research_company(
                    company=company,
                    role=job.title,
                    cache_dir=DEFAULT_COMPANY_RESEARCH,
                    cache=not args.no_cache,
                )
            except RuntimeError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1
        payload = generate_brief(
            job_file,
            store,
            embedder,
            memory,
            top_k=args.top_k,
            write_contract=args.write_contract,
            job_text=job.text,
            job_title=job.title,
            job_company=job.company,
            job_url=job.url,
            job_source_type=job.source_type,
            job_cached_path=job.cached_path,
            web_research=web_sources,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "research-company":
        try:
            payload = research_company(
                company=args.company,
                role=args.role,
                num_results=args.num_results,
                cache_dir=DEFAULT_COMPANY_RESEARCH,
                cache=not args.no_cache,
            )
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "evaluate":
        payload = evaluate_retrieval(args.queries, store, embedder, k=args.k)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    return 2
