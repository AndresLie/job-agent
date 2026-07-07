from __future__ import annotations

import re
from pathlib import Path

from .answer import build_citations, nvidia_chat
from .contracts import validate_brief, write_json_contract
from .documents import clean_text
from .embeddings import Embedder
from .memory import MemoryStore
from .vector_store import JsonVectorStore, keyword_tokens


RESUME_CATEGORIES = {"resume"}
SUPPORTING_CATEGORIES = {"projects", "experience"}


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
    resume_hits = store.query(comparison_text, embedder, top_k, categories=RESUME_CATEGORIES)
    supporting_hits = store.query(comparison_text, embedder, top_k, categories=SUPPORTING_CATEGORIES)
    memories = [item.summary for item in memory.retrieve(comparison_text, k=5)]
    job_terms = skill_terms(comparison_text)
    resume_terms = collect_terms(resume_hits)
    supporting_terms = collect_terms(supporting_hits)

    matched = sorted(job_terms & resume_terms)
    hidden_terms = sorted((job_terms - resume_terms) & supporting_terms)
    gaps = sorted(job_terms - resume_terms - supporting_terms)[:8]
    missing_from_cv = sorted(job_terms - resume_terms)
    fit_score = score_resume_match(job_terms, resume_terms, resume_hits)
    brutal_assessment = build_brutal_assessment(fit_score, matched, hidden_terms, gaps, resume_hits, supporting_hits)
    llm_review = generate_brutal_llm_review(
        job_title=job_title or infer_title(job_text, job_path),
        job_text=job_text,
        fit_score=fit_score,
        matched=matched,
        hidden_terms=hidden_terms,
        gaps=gaps,
        resume_hits=resume_hits,
        supporting_hits=supporting_hits,
    )
    all_citation_hits = resume_hits + supporting_hits
    payload = {
        "job_title": job_title or infer_title(job_text, job_path),
        "job_input": {
            "source_type": job_source_type,
            "company": job_company,
            "url": job_url,
            "cached_path": job_cached_path,
        },
        "fit_score": fit_score,
        "cv_match": {
            "score": fit_score,
            "verdict": score_verdict(fit_score),
            "matched_terms": matched,
            "missing_from_cv": missing_from_cv[:12],
            "feedback": build_cv_feedback(fit_score, matched, missing_from_cv, resume_hits),
        },
        "matched_evidence": build_matched_evidence(resume_hits, matched, memories),
        "hidden_evidence": build_hidden_evidence(supporting_hits, hidden_terms),
        "skill_gaps": gaps,
        "recommended_actions": build_actions(gaps, hidden_terms, supporting_hits, resume_hits),
        "brutal_assessment": brutal_assessment,
        "llm_brutal_review": llm_review,
        "citations": build_citations(all_citation_hits),
        "web_research": web_research or [],
        "confidence": confidence_score(resume_hits, supporting_hits),
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


def skill_terms(text: str) -> set[str]:
    terms = {term for term in keyword_tokens(text) if is_skill_like(term)}
    return terms or keyword_tokens(text)


def collect_terms(hits: list[dict]) -> set[str]:
    terms = set()
    for hit in hits:
        terms |= skill_terms(hit["text"])
    return terms


def score_resume_match(job_terms: set[str], resume_terms: set[str], resume_hits: list[dict]) -> int:
    if not resume_hits or not job_terms:
        return 0
    return min(100, round((len(job_terms & resume_terms) / len(job_terms)) * 100))


def score_verdict(score: int) -> str:
    if score >= 75:
        return "strong_cv_match"
    if score >= 50:
        return "partial_cv_match"
    if score >= 25:
        return "weak_cv_match"
    return "not_competitive_from_cv"


def build_cv_feedback(score: int, matched: list[str], missing_from_cv: list[str], resume_hits: list[dict]) -> str:
    if not resume_hits:
        return "No resume/CV evidence was indexed, so the JD-to-CV score is 0. Add a resume or CV under a resume/ folder."
    if score >= 75:
        return "The current CV covers most visible role requirements. Tighten the bullets around quantified impact and the strongest matched terms."
    if score >= 50:
        return (
            "The CV shows a partial match, but it does not sell the role strongly enough. "
            f"Matched terms: {', '.join(matched[:6]) or 'none'}. Missing from CV: {', '.join(missing_from_cv[:6]) or 'none'}."
        )
    if score >= 25:
        return (
            "The CV is weak for this JD as written. It may contain adjacent experience, but too many required terms are absent or implicit."
        )
    return "The CV is not competitive for this JD as written. Do not rely on generic AI/DS wording; add concrete, role-specific evidence or target a closer role."


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


def build_hidden_evidence(hits: list[dict], hidden_terms: list[str]) -> list[dict]:
    evidence = []
    for hit in hits[:6]:
        terms = sorted(skill_terms(hit["text"]) & set(hidden_terms))[:8]
        if not terms:
            continue
        evidence.append(
            {
                "source": hit["filename"],
                "category": hit.get("category", "general"),
                "source_path": hit.get("source_path", hit["filename"]),
                "chunk_index": hit["chunk_index"],
                "terms_to_add_to_cv": terms,
                "summary": re.sub(r"\s+", " ", hit["text"]).strip()[:320],
            }
        )
    return evidence


def build_actions(
    gaps: list[str],
    hidden_terms: list[str],
    supporting_hits: list[dict],
    resume_hits: list[dict],
) -> list[str]:
    actions = []
    if not resume_hits:
        actions.append("Add a real CV or resume under the resume/ folder before trusting any score.")
    if hidden_terms and supporting_hits:
        actions.append(
            "Rewrite the CV to include project/experience evidence for: " + ", ".join(hidden_terms[:6]) + "."
        )
    if gaps:
        actions.append("Build or document stronger evidence before applying for roles requiring: " + ", ".join(gaps[:6]) + ".")
    if not hidden_terms and not gaps and resume_hits:
        actions.append("Use the strongest CV evidence directly in the cover letter and recruiter summary.")
    actions.append("After editing the CV, rerun ingest and brief so the score reflects the new resume wording.")
    return actions


def build_brutal_assessment(
    fit_score: int,
    matched: list[str],
    hidden_terms: list[str],
    gaps: list[str],
    resume_hits: list[dict],
    supporting_hits: list[dict],
) -> str:
    if not resume_hits:
        return "Brutal truth: there is no indexed CV/resume, so the system cannot claim you fit this JD."
    if fit_score < 25 and not hidden_terms:
        return (
            "Brutal truth: your indexed CV, projects, and experience are nowhere near this JD based on available evidence. "
            "Apply only if you can add real missing evidence, not just rewrite wording."
        )
    if fit_score < 40 and hidden_terms:
        return (
            "Brutal truth: your CV is not competitive as written, but your projects/experience contain some evidence you failed to surface. "
            "The next move is a CV rewrite, not pretending the current CV is enough."
        )
    if fit_score < 65:
        return (
            "Brutal truth: this is a stretch. You have some relevant evidence, but the CV leaves too many role requirements weak or missing."
        )
    if gaps:
        return "Brutal truth: the CV is directionally good, but the remaining gaps could cost interviews for stricter screens."
    return "Brutal truth: the CV is a credible match from the indexed evidence. Improve specificity and quantified impact."


def confidence_score(resume_hits: list[dict], supporting_hits: list[dict]) -> float:
    if resume_hits and supporting_hits:
        return 0.82
    if resume_hits:
        return 0.68
    if supporting_hits:
        return 0.48
    return 0.25


def generate_brutal_llm_review(
    *,
    job_title: str,
    job_text: str,
    fit_score: int,
    matched: list[str],
    hidden_terms: list[str],
    gaps: list[str],
    resume_hits: list[dict],
    supporting_hits: list[dict],
) -> str | None:
    resume_context = format_hits_for_prompt(resume_hits[:4])
    supporting_context = format_hits_for_prompt(supporting_hits[:5])
    return nvidia_chat(
        system_prompt=(
            "You are a brutally honest career reviewer for data scientist and AI roles. "
            "Use only the supplied JD, CV evidence, and project/experience evidence. "
            "Do not flatter the candidate. If the evidence is insufficient, say so clearly. "
            "Separate current CV fit from recommendations to improve the CV."
        ),
        user_prompt=(
            f"Job title: {job_title}\n"
            f"JD excerpt:\n{job_text[:3500]}\n\n"
            f"Deterministic JD-to-CV score: {fit_score}/100\n"
            f"CV matched terms: {', '.join(matched) or 'none'}\n"
            f"Terms found in projects/experience but missing from CV: {', '.join(hidden_terms) or 'none'}\n"
            f"Terms with no evidence: {', '.join(gaps) or 'none'}\n\n"
            f"CV evidence:\n{resume_context or 'No CV evidence.'}\n\n"
            f"Project/experience evidence:\n{supporting_context or 'No project/experience evidence.'}\n\n"
            "Return concise feedback with: verdict, why, what to add to the CV, and whether applying is realistic."
        ),
    )


def format_hits_for_prompt(hits: list[dict]) -> str:
    lines = []
    for index, hit in enumerate(hits, start=1):
        text = re.sub(r"\s+", " ", hit["text"]).strip()[:700]
        lines.append(f"[{index}] {hit.get('category', 'general')}/{hit['filename']} chunk {hit['chunk_index']}: {text}")
    return "\n".join(lines)
