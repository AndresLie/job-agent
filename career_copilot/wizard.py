from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .brief import generate_brief
from .documents import chunk_document, discover_documents, load_document
from .embeddings import Embedder
from .job_input import resolve_job_input
from .memory import MemoryStore
from .vector_store import JsonVectorStore, clear_storage
from .web_research import research_company


InputFn = Callable[[str], str]


def parse_yes_no(value: str, default: bool = False) -> bool:
    text = value.strip().casefold()
    if not text:
        return default
    if text in {"y", "yes", "true", "1"}:
        return True
    if text in {"n", "no", "false", "0"}:
        return False
    raise ValueError("Expected yes or no.")


def parse_choice(value: str, choices: set[str], default: str) -> str:
    text = value.strip().casefold()
    if not text:
        return default
    if text not in choices:
        raise ValueError(f"Expected one of: {', '.join(sorted(choices))}.")
    return text


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def ingest_folder(source: Path, store: JsonVectorStore, embedder: Embedder, rebuild: bool, storage: Path) -> tuple[int, int]:
    if rebuild:
        clear_storage(storage)
        store.records = []
    docs = discover_documents(source)
    root = source if source.is_dir() else source.parent
    chunks = []
    for doc in docs:
        chunks.extend(chunk_document(doc, load_document(doc), root=root))
    store.upsert_chunks(chunks, embedder)
    return len(docs), len(chunks)


def run_wizard(
    *,
    project_root: Path,
    data_dir: Path,
    storage: Path,
    jobs_dir: Path,
    company_research_dir: Path,
    store: JsonVectorStore,
    embedder: Embedder,
    memory: MemoryStore,
    input_fn: InputFn = input,
) -> dict:
    print("AI Job Copilot Wizard")
    rag_text = input_fn(f"RAG folder [{data_dir}]: ").strip()
    rag_folder = Path(rag_text) if rag_text else data_dir
    rebuild = parse_yes_no(input_fn("Rebuild index? [Y/n]: "), default=True)
    doc_count, chunk_count = ingest_folder(rag_folder, store, embedder, rebuild, storage)
    print(f"Indexed {chunk_count} chunks from {doc_count} documents.")

    memory_text = input_fn("Memory note to remember now [optional]: ").strip()
    if memory_text:
        memory.add(memory_text, tags=["wizard"])
        print("Remembered.")

    mode = parse_choice(input_fn("JD input mode: text, file, or url [text]: "), {"text", "file", "url"}, "text")
    company = input_fn("Company name [optional]: ").strip() or None
    if mode == "file":
        job = resolve_job_input(job_file=Path(input_fn("JD file path: ").strip()), company=company, cache_dir=jobs_dir)
    elif mode == "url":
        job = resolve_job_input(job_url=input_fn("JD URL: ").strip(), company=company, cache_dir=jobs_dir)
    else:
        print("Paste JD text. End with a single line containing only END.")
        lines = []
        while True:
            line = input_fn("")
            if line.strip() == "END":
                break
            lines.append(line)
        job = resolve_job_input(job_text="\n".join(lines), company=company, cache_dir=jobs_dir)

    research = []
    if parse_yes_no(input_fn("Run Exa company research? [y/N]: "), default=False):
        company_name = job.company or company
        if not company_name:
            raise ValueError("Company name is required for web research.")
        research = research_company(
            company=company_name,
            role=job.title,
            cache_dir=company_research_dir,
            cache=True,
        )

    payload = generate_brief(
        None,
        store,
        embedder,
        memory,
        job_text=job.text,
        job_title=job.title,
        job_company=job.company,
        job_url=job.url,
        job_source_type=job.source_type,
        job_cached_path=job.cached_path,
        web_research=research,
    )
    output = project_root / "outputs" / "latest_brief.json"
    write_json(output, payload)
    print(f"Wrote {output}")
    return payload


def run_demo(
    *,
    project_root: Path,
    store: JsonVectorStore,
    embedder: Embedder,
    memory: MemoryStore,
    storage: Path,
) -> dict:
    examples = project_root / "examples"
    demo_storage = project_root / "storage" / "demo"
    demo_store = JsonVectorStore(demo_storage / "vector_store.json")
    demo_memory = MemoryStore(demo_storage / "memory.json")
    docs, chunks = ingest_folder(examples, demo_store, embedder, rebuild=True, storage=demo_storage)
    demo_memory.add("The candidate prefers reliable AI systems with evaluation and retrieval.", tags=["demo"])
    job_path = examples / "jobs" / "sample_job.md"
    payload = generate_brief(job_path, demo_store, embedder, demo_memory)
    output = project_root / "outputs" / "demo_brief.json"
    write_json(output, payload)
    print(f"Demo indexed {chunks} chunks from {docs} documents.")
    print(f"Wrote {output}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload
