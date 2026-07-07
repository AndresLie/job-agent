from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .brief import generate_brief
from .config import load_env_file
from .documents import chunk_document, discover_documents, load_document
from .embeddings import build_embedder
from .job_input import resolve_job_input
from .memory import MemoryStore
from .vector_store import JsonVectorStore
from .web_research import research_company
from .wizard import write_json


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = PACKAGE_ROOT.parents[0]
TEMPLATES = Jinja2Templates(directory=str(PACKAGE_ROOT / "templates"))


def create_app(project_root: Path | None = None) -> FastAPI:
    root = (project_root or DEFAULT_PROJECT_ROOT).resolve()
    load_env_file(root / ".env")
    app = FastAPI(title="AI Job Copilot")
    static_dir = PACKAGE_ROOT / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    app.state.project_root = root
    app.state.latest_brief = load_latest(root)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/latest")
    def latest() -> JSONResponse:
        payload = app.state.latest_brief or load_latest(root) or {}
        return JSONResponse(payload)

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        return render(request, root=root)

    @app.post("/review", response_class=HTMLResponse)
    def review(
        request: Request,
        rag_folder: str = Form("data/raw"),
        rebuild_index: bool = Form(True),
        job_text: str = Form(""),
        job_url: str = Form(""),
        company: str = Form(""),
        research_company_enabled: bool = Form(False),
        top_k: int = Form(8),
    ) -> HTMLResponse:
        form = {
            "rag_folder": rag_folder,
            "rebuild_index": rebuild_index,
            "job_text": job_text,
            "job_url": job_url,
            "company": company,
            "research_company_enabled": research_company_enabled,
            "top_k": top_k,
        }
        try:
            payload, notice = run_review(
                root=root,
                rag_folder=rag_folder,
                rebuild_index=rebuild_index,
                job_text=job_text,
                job_url=job_url,
                company=company,
                research_company_enabled=research_company_enabled,
                top_k=top_k,
            )
            app.state.latest_brief = payload
            return render(request, root=root, payload=payload, notice=notice, form=form)
        except Exception as exc:
            return render(request, root=root, error=str(exc), form=form, status_code=400)

    return app


def run_review(
    *,
    root: Path,
    rag_folder: str,
    rebuild_index: bool,
    job_text: str,
    job_url: str,
    company: str,
    research_company_enabled: bool,
    top_k: int,
) -> tuple[dict[str, Any], str]:
    data_dir = root / "data" / "raw"
    storage = root / "storage"
    index_path = storage / "vector_store.json"
    memory_path = storage / "memory.json"
    jobs_dir = data_dir / "jobs"
    company_research_dir = data_dir / "company_research"
    output = root / "outputs" / "latest_brief.json"

    embedder = build_embedder("hashing")
    store = JsonVectorStore(index_path)
    memory = MemoryStore(memory_path)

    notices = []
    if rebuild_index:
        source = resolve_local_path(rag_folder or "data/raw", root)
        if not source.exists():
            raise ValueError(f"RAG folder does not exist: {source}")
        store.reset()
        docs = discover_documents(source)
        chunks = []
        doc_root = source if source.is_dir() else source.parent
        for doc in docs:
            chunks.extend(chunk_document(doc, load_document(doc), root=doc_root))
        store.upsert_chunks(chunks, embedder)
        notices.append(f"Indexed {len(chunks)} chunks from {len(docs)} documents.")
    elif not store.records:
        raise ValueError("No existing index found. Enable rebuild index and choose a RAG folder.")

    job_text = job_text.strip()
    job_url = job_url.strip()
    if bool(job_text) == bool(job_url):
        raise ValueError("Provide exactly one JD input: pasted text or job URL.")
    job = resolve_job_input(
        job_text=job_text or None,
        job_url=job_url or None,
        company=company.strip() or None,
        cache_dir=jobs_dir,
        cache=True,
    )

    web_sources = []
    if research_company_enabled:
        company_name = job.company or company.strip()
        if not company_name:
            raise ValueError("Company name is required for company research.")
        web_sources = research_company(
            company=company_name,
            role=job.title,
            cache_dir=company_research_dir,
            cache=True,
        )
        notices.append(f"Fetched {len(web_sources)} company research sources.")

    payload = generate_brief(
        None,
        store,
        embedder,
        memory,
        top_k=top_k,
        job_text=job.text,
        job_title=job.title,
        job_company=job.company,
        job_url=job.url,
        job_source_type=job.source_type,
        job_cached_path=job.cached_path,
        web_research=web_sources,
    )
    write_json(output, payload)
    notices.append(f"Wrote {output}")
    return payload, " ".join(notices)


def resolve_local_path(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def load_latest(root: Path) -> dict[str, Any] | None:
    path = root / "outputs" / "latest_brief.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict) or "cv_jd_review" not in payload:
        return None
    return payload


def render(
    request: Request,
    *,
    root: Path,
    payload: dict[str, Any] | None = None,
    error: str | None = None,
    notice: str | None = None,
    form: dict[str, Any] | None = None,
    status_code: int = 200,
) -> HTMLResponse:
    default_form = {
        "rag_folder": str(root / "data" / "raw"),
        "rebuild_index": True,
        "job_text": "",
        "job_url": "",
        "company": "",
        "research_company_enabled": False,
        "top_k": 8,
    }
    if form:
        default_form.update(form)
    context = {
        "request": request,
        "form": default_form,
        "payload": payload,
        "error": error,
        "notice": notice,
        "latest": payload or load_latest(root),
    }
    return TEMPLATES.TemplateResponse(request, "review.html", context, status_code=status_code)
