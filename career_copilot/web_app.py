from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .brief import generate_brief
from .config import load_env_file
from .documents import chunk_document, discover_documents, load_document
from .embeddings import build_embedder
from .job_input import resolve_job_input
from .memory import MemoryStore
from .rubric import analyze_job_requirements
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

    @app.get("/history", response_class=HTMLResponse)
    def history(request: Request) -> HTMLResponse:
        return render(request, root=root, history=list_review_runs(root), show_history=True)

    @app.get("/history/{run_id}", response_class=HTMLResponse)
    def history_detail(request: Request, run_id: str) -> HTMLResponse:
        try:
            payload = load_review_run(root, run_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return render(request, root=root, payload=payload, notice=f"Loaded review {run_id}.")

    @app.post("/preview-job", response_class=HTMLResponse)
    def preview_job(
        request: Request,
        rag_folder: str = Form("data/raw"),
        rebuild_index: bool = Form(True),
        job_text: str = Form(""),
        job_url: str = Form(""),
        company: str = Form(""),
        research_company_enabled: bool = Form(False),
        memory_note: str = Form(""),
        memory_tags: str = Form(""),
        memory_query: str = Form(""),
        top_k: int = Form(8),
    ) -> HTMLResponse:
        form = {
            "rag_folder": rag_folder,
            "rebuild_index": rebuild_index,
            "job_text": job_text,
            "job_url": job_url,
            "company": company,
            "research_company_enabled": research_company_enabled,
            "memory_note": memory_note,
            "memory_tags": memory_tags,
            "memory_query": memory_query,
            "top_k": top_k,
        }
        try:
            if not job_url.strip():
                raise ValueError("Enter a job URL to preview.")
            job = resolve_job_input(
                job_url=job_url.strip(),
                company=company.strip() or None,
                cache_dir=root / "data" / "raw" / "jobs",
                cache=True,
            )
            form.update(
                {
                    "job_text": job.text,
                    "job_url": "",
                    "company": job.company or company,
                }
            )
            preview = build_job_preview(job, job_url.strip())
            notice = "Extracted JD preview. Edit the text if needed, then run review."
            return render(request, root=root, form=form, preview=preview, notice=notice)
        except Exception as exc:
            return render(request, root=root, error=str(exc), form=form, status_code=400)

    @app.post("/review", response_class=HTMLResponse)
    def review(
        request: Request,
        rag_folder: str = Form("data/raw"),
        rebuild_index: bool = Form(True),
        job_text: str = Form(""),
        job_url: str = Form(""),
        company: str = Form(""),
        research_company_enabled: bool = Form(False),
        memory_note: str = Form(""),
        memory_tags: str = Form(""),
        memory_query: str = Form(""),
        top_k: int = Form(8),
    ) -> HTMLResponse:
        form = {
            "rag_folder": rag_folder,
            "rebuild_index": rebuild_index,
            "job_text": job_text,
            "job_url": job_url,
            "company": company,
            "research_company_enabled": research_company_enabled,
            "memory_note": memory_note,
            "memory_tags": memory_tags,
            "memory_query": memory_query,
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
                memory_note=memory_note,
                memory_tags=memory_tags,
                memory_query=memory_query,
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
    memory_note: str,
    memory_tags: str,
    memory_query: str,
    top_k: int,
) -> tuple[dict[str, Any], str]:
    data_dir = root / "data" / "raw"
    storage = root / "storage"
    index_path = storage / "vector_store.json"
    memory_path = storage / "memory.json"
    jobs_dir = data_dir / "jobs"
    company_research_dir = data_dir / "company_research"
    latest_output = root / "outputs" / "latest_brief.json"
    run_output = review_run_path(root)

    embedder = build_embedder("hashing")
    store = JsonVectorStore(index_path)
    memory = MemoryStore(memory_path)

    notices = []
    diagnostics: dict[str, Any] = {
        "index_rebuilt": bool(rebuild_index),
        "indexed_documents": None,
        "indexed_chunks": None,
        "job_extraction_method": None,
        "job_text_chars": None,
        "web_research_status": "disabled",
        "web_research_sources": 0,
        "memory_added": False,
        "memory_recall": [],
        "result_path": str(run_output),
        "latest_path": str(latest_output),
    }
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
        diagnostics["indexed_documents"] = len(docs)
        diagnostics["indexed_chunks"] = len(chunks)
        notices.append(f"Indexed {len(chunks)} chunks from {len(docs)} documents.")
    elif not store.records:
        raise ValueError("No existing index found. Enable rebuild index and choose a RAG folder.")
    else:
        diagnostics["indexed_chunks"] = len(store.records)

    memory_note = memory_note.strip()
    if memory_note:
        tags = [tag.strip() for tag in memory_tags.split(",") if tag.strip()]
        diagnostics["memory_added"] = memory.add(memory_note, tags=tags, source="web")
        notices.append("Memory saved." if diagnostics["memory_added"] else "Memory already existed.")

    memory_query = memory_query.strip()
    if memory_query:
        diagnostics["memory_recall"] = [
            {"summary": item.summary, "tags": item.tags, "created_at": item.created_at}
            for item in memory.retrieve(memory_query, k=5)
        ]

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
    diagnostics["job_extraction_method"] = job.extraction_method
    diagnostics["job_text_chars"] = len(job.text)

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
        diagnostics["web_research_status"] = "used"
        diagnostics["web_research_sources"] = len(web_sources)
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
        diagnostics=diagnostics,
    )
    write_json(run_output, payload)
    write_json(latest_output, payload)
    notices.append(f"Wrote {run_output}")
    return payload, " ".join(notices)


def resolve_local_path(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve()
    if not is_relative_to(resolved, root):
        raise ValueError("RAG folder must stay inside the project directory.")
    blocked = {".git", "storage", "outputs", "__pycache__"}
    rel_parts = {part.casefold() for part in resolved.relative_to(root).parts}
    if rel_parts & blocked:
        raise ValueError("RAG folder cannot point at project metadata, storage, or output folders.")
    return resolved


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def review_run_path(root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / "outputs" / "runs" / f"review-{timestamp}-{uuid.uuid4().hex[:8]}.json"


def load_latest(root: Path) -> dict[str, Any] | None:
    path = root / "outputs" / "latest_brief.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict) or "cv_jd_review" not in payload or "diagnostics" not in payload:
        return None
    return payload


def build_job_preview(job: Any, source_url: str | None = None) -> dict[str, Any]:
    requirements = analyze_job_requirements(job.text)
    return {
        "title": job.title,
        "company": job.company,
        "location": infer_job_location(job.text),
        "method": job.extraction_method,
        "chars": len(job.text),
        "source_url": source_url or job.url,
        "role_family": requirements.get("role_family", "general_ai_data"),
        "required": requirements.get("required", [])[:12],
        "preferred": requirements.get("preferred", [])[:8],
        "responsibilities": requirements.get("responsibilities", [])[:8],
    }


def infer_job_location(text: str) -> str:
    for line in text.splitlines()[:40]:
        stripped = line.strip()
        if not stripped:
            continue
        match = re.match(r"(?i)^(location|job location|office|work location)\s*[:\-]\s*(.+)$", stripped)
        if match:
            return match.group(2).strip()[:80]
        if re.search(r"(?i)\b(remote|hybrid|onsite|taiwan|taipei|hsinchu|singapore|malaysia|united states)\b", stripped):
            return stripped[:80]
    return "Not detected"


def list_review_runs(root: Path) -> list[dict[str, Any]]:
    runs = []
    directory = root / "outputs" / "runs"
    if not directory.exists():
        return runs
    for path in sorted(directory.glob("review-*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        payload = read_review_payload(path)
        if not payload:
            continue
        verdict = payload.get("application_verdict") or {}
        diagnostics = payload.get("diagnostics") or {}
        runs.append(
            {
                "run_id": path.name,
                "job_title": payload.get("job_title", "Untitled"),
                "role_family": payload.get("role_family", "unknown"),
                "fit_score": payload.get("fit_score", "n/a"),
                "verdict": verdict.get("label", "unknown"),
                "extraction_method": diagnostics.get("job_extraction_method", "unknown"),
                "modified_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            }
        )
    return runs


def load_review_run(root: Path, run_id: str) -> dict[str, Any]:
    if Path(run_id).name != run_id:
        raise ValueError("Invalid review id.")
    path = (root / "outputs" / "runs" / run_id).resolve()
    runs_root = (root / "outputs" / "runs").resolve()
    if not is_relative_to(path, runs_root) or not path.name.startswith("review-") or path.suffix != ".json":
        raise ValueError("Invalid review id.")
    payload = read_review_payload(path)
    if not payload:
        raise ValueError("Review not found.")
    return payload


def read_review_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict) or "cv_jd_review" not in payload or "diagnostics" not in payload:
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
    preview: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
    show_history: bool = False,
    status_code: int = 200,
) -> HTMLResponse:
    default_form = {
        "rag_folder": str(root / "data" / "raw"),
        "rebuild_index": True,
        "job_text": "",
        "job_url": "",
        "company": "",
        "research_company_enabled": False,
        "memory_note": "",
        "memory_tags": "",
        "memory_query": "",
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
        "preview": preview,
        "history": history or [],
        "show_history": show_history,
    }
    return TEMPLATES.TemplateResponse(request, "review.html", context, status_code=status_code)
