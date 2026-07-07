from __future__ import annotations

import re
from pathlib import Path

from .answer import build_citations
from .contracts import validate_brief, write_json_contract
from .documents import clean_text
from .embeddings import Embedder
from .memory import MemoryStore
from .vector_store import JsonVectorStore, keyword_tokens


def generate_brief(
    job_path: Path | None,
    store: JsonVectorStore,
    embedder: Embedder,
    memory: MemoryStore,
    top_k: int = 8,
    write_contract: bool = False,
    job_text: str | None = None,
    job_title: str | None = None,
    job_company: str = "",
    job_url: str | None = None,
    job_source_type: str = "file",
    job_cached_path: str | None = None,
    web_research: list[dict] | None = None,
) -> dict:
    if job_text is None:
        if job_path is None:
            raise ValueError("job_path or job_text is required")
        job_text = job_path.read_text(encoding="utf-8", errors="ignore")
    job_text = clean_text(job_text)
    research_text = "\n".join(
        " ".join([source.get("title", ""), source.get("summary", ""), " ".join(source.get("highlights", []))])
        for source in (web_research or [])
    )
    comparison_text = clean_text(job_text + "\n\n" + research_text)
    hits = store.query(comparison_text, embedder, top_k)
    memories = [item.summary for item in memory.retrieve(comparison_text, k=5)]
    job_terms = keyword_tokens(comparison_text)
    evidence_terms = set()
    for hit in hits:
        evidence_terms |= keyword_tokens(hit["text"])

    matched = sorted(job_terms & evidence_terms)
    gaps = sorted(term for term in job_terms - evidence_terms if is_skill_like(term))[:8]
    fit_score = min(100, int((len(matched) / max(len(job_terms), 1)) * 140))
    payload = {
        "job_title": job_title or infer_title(job_text, job_path),
        "job_input": {
            "source_type": job_source_type,
            "company": job_company,
            "url": job_url,
            "cached_path": job_cached_path,
        },
        "fit_score": fit_score,
        "matched_evidence": build_matched_evidence(hits, matched, memories),
        "skill_gaps": gaps,
        "recommended_actions": build_actions(gaps, hits),
        "citations": build_citations(hits),
        "web_research": web_research or [],
        "confidence": 0.78 if hits else 0.35,
    }
    ok, reason = validate_brief(payload)
    if not ok:
        raise ValueError(reason)
    if write_contract:
        write_json_contract(payload)
    return payload


def infer_title(job_text: str, path: Path | None) -> str:
    for line in job_text.splitlines():
        stripped = line.strip("# ").strip()
        if stripped and len(stripped) < 90:
            return stripped
    if path is None:
        return "Job Description"
    return path.stem.replace("_", " ").replace("-", " ").title()


def is_skill_like(term: str) -> bool:
    return term in {
        "python",
        "sql",
        "rag",
        "retrieval",
        "evaluation",
        "ml",
        "machine",
        "learning",
        "llm",
        "agent",
        "agents",
        "pytorch",
        "spark",
        "airflow",
        "docker",
        "aws",
        "kubernetes",
        "statistics",
        "experiment",
        "experimentation",
    }


def build_matched_evidence(hits: list[dict], matched: list[str], memories: list[str]) -> list[dict]:
    evidence = []
    for hit in hits[:5]:
        terms = sorted(keyword_tokens(hit["text"]) & set(matched))[:8]
        evidence.append(
            {
                "source": hit["filename"],
                "category": hit.get("category", "general"),
                "source_path": hit.get("source_path", hit["filename"]),
                "chunk_index": hit["chunk_index"],
                "matched_terms": terms,
                "summary": re.sub(r"\s+", " ", hit["text"]).strip()[:260],
            }
        )
    for item in memories[:2]:
        evidence.append({"source": "memory", "chunk_index": None, "matched_terms": [], "summary": item})
    return evidence


def build_actions(gaps: list[str], hits: list[dict]) -> list[str]:
    actions = []
    if gaps:
        actions.append("Add one project note or resume bullet that directly demonstrates: " + ", ".join(gaps[:4]) + ".")
    if hits:
        actions.append("Use the strongest cited project evidence in the cover letter and interview story.")
    actions.append("Run evaluate after adding new evidence to confirm retrieval quality remains stable.")
    return actions
