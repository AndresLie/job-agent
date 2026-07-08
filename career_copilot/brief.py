from __future__ import annotations

import re
from pathlib import Path

from .answer import build_citations, has_llm_config, nvidia_chat
from .contracts import validate_brief, write_json_contract
from .documents import clean_text
from .embeddings import Embedder
from .memory import MemoryStore
from .rubric import analyze_evidence_depth, analyze_job_requirements, extract_skills, known_skill_terms, score_with_rubric
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
    diagnostics: dict | None = None,
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
    jd_comparison_text = clean_text(job_text)
    retrieval_text = clean_text(job_text + "\n\n" + research_text)
    resume_hits = store.query(jd_comparison_text, embedder, top_k, categories=RESUME_CATEGORIES)
    supporting_hits = store.query(jd_comparison_text, embedder, top_k, categories=SUPPORTING_CATEGORIES)
    memories = [item.summary for item in memory.retrieve(retrieval_text, k=5)]
    requirements = analyze_job_requirements(jd_comparison_text)
    evidence_depth = analyze_evidence_depth(requirements, resume_hits, supporting_hits)
    rubric_result = score_with_rubric(requirements, evidence_depth, resume_hits)
    job_terms = set(requirements["required"]) | set(requirements["preferred"])
    resume_terms = collect_terms(resume_hits)
    supporting_terms = collect_terms(supporting_hits)

    matched = rubric_result["matched_terms"]
    hidden_terms = rubric_result["hidden_terms"] or sorted((job_terms - resume_terms) & supporting_terms)
    gaps = rubric_result["skill_gaps"][:8]
    missing_from_cv = rubric_result["missing_from_cv"]
    weak_evidence = rubric_result["weak_evidence"]
    fit_score = rubric_result["fit_score"]
    application_verdict = build_application_verdict(fit_score, hidden_terms, gaps, weak_evidence, resume_hits)
    cv_rewrite_suggestions = build_cv_rewrite_suggestions(supporting_hits, hidden_terms)
    score_explanations = build_score_explanations(
        requirements=requirements,
        scoring_breakdown=rubric_result["scoring_breakdown"],
        evidence_depth=evidence_depth,
        resume_hits=resume_hits,
        matched=matched,
        weak_evidence=weak_evidence,
    )
    brutal_assessment = build_brutal_assessment(fit_score, matched, hidden_terms, gaps, resume_hits, supporting_hits)
    llm_review = generate_brutal_llm_review(
        job_title=job_title or infer_title(job_text, job_path),
        job_text=job_text,
        fit_score=fit_score,
        application_verdict=application_verdict,
        cv_rewrite_suggestions=cv_rewrite_suggestions,
        requirements=requirements,
        evidence_depth=evidence_depth,
        scoring_breakdown=rubric_result["scoring_breakdown"],
        weak_evidence=weak_evidence,
        matched=matched,
        hidden_terms=hidden_terms,
        gaps=gaps,
        resume_hits=resume_hits,
        supporting_hits=supporting_hits,
    )
    diagnostics_payload = dict(diagnostics or {})
    diagnostics_payload.update(
        {
            "llm_status": "used" if llm_review else ("configured_but_unavailable" if has_llm_config() else "not_configured"),
            "memory_hits": len(memories),
            "memory_summaries": memories[:5],
            "resume_chunks_considered": len(resume_hits),
            "supporting_chunks_considered": len(supporting_hits),
            "credible_cv_terms": rubric_result.get("credible_matched_terms", []),
        }
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
        "role_family": requirements["role_family"],
        "jd_requirements": requirements,
        "evidence_depth": evidence_depth,
        "scoring_breakdown": rubric_result["scoring_breakdown"],
        "score_explanations": score_explanations,
        "weak_evidence": weak_evidence,
        "cv_jd_review": {
            "score": fit_score,
            "verdict": score_verdict(fit_score),
            "reason": build_cv_feedback(fit_score, matched, missing_from_cv, resume_hits),
            "matched_terms": matched,
            "missing_from_cv": missing_from_cv[:12],
            "weak_evidence": weak_evidence,
            "scoring_breakdown": rubric_result["scoring_breakdown"],
        },
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
        "application_verdict": application_verdict,
        "cv_rewrite_suggestions": cv_rewrite_suggestions,
        "recommended_actions": build_actions(gaps, hidden_terms, supporting_hits, resume_hits),
        "brutal_assessment": brutal_assessment,
        "llm_brutal_review": llm_review,
        "citations": build_citations(all_citation_hits),
        "web_research": web_research or [],
        "confidence": confidence_score(resume_hits, supporting_hits),
        "diagnostics": diagnostics_payload,
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
    return term in known_skill_terms()


def skill_terms(text: str) -> set[str]:
    normalized = re.sub(r"\s+", " ", text.casefold()).strip()
    terms = set(extract_skills(normalized))
    terms |= {term for term in keyword_tokens(text) if is_skill_like(term)}
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


def build_application_verdict(
    fit_score: int,
    hidden_terms: list[str],
    gaps: list[str],
    weak_evidence: list[str],
    resume_hits: list[dict],
) -> dict:
    if not resume_hits:
        return {
            "label": "not_competitive",
            "apply_now": False,
            "risk_level": "high",
            "reason": "No indexed CV/resume evidence was found, so the application is not defensible yet.",
        }
    if fit_score < 35:
        if hidden_terms:
            return {
                "label": "weak_match",
                "apply_now": False,
                "risk_level": "high",
                "reason": "The current CV is very weak for this JD, though projects or experience contain evidence that could support a rewrite.",
            }
        return {
            "label": "not_competitive",
            "apply_now": False,
            "risk_level": "high",
            "reason": "The current CV and supporting evidence do not cover enough of the JD requirements.",
        }
    if fit_score < 55:
        return {
            "label": "weak_match",
            "apply_now": False,
            "risk_level": "high",
            "reason": "The current CV has some relevant signals, but it is unlikely to pass a strict screen without a rewrite.",
        }
    if fit_score < 75 or len(weak_evidence) >= 3:
        return {
            "label": "stretch",
            "apply_now": True,
            "risk_level": "medium",
            "reason": "The CV has a plausible base match, but gaps and missing evidence make this a stretch application.",
        }
    if len(gaps) >= 3:
        return {
            "label": "stretch",
            "apply_now": True,
            "risk_level": "medium",
            "reason": "The CV has strong overlap, but several uncovered requirements remain risky.",
        }
    return {
        "label": "strong_match",
        "apply_now": True,
        "risk_level": "low",
        "reason": "The current CV covers most visible JD requirements from indexed evidence.",
    }


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


def build_cv_rewrite_suggestions(hits: list[dict], hidden_terms: list[str]) -> list[dict]:
    suggestions = []
    seen_sources = set()
    for hit in hits:
        terms = sorted(skill_terms(hit["text"]) & set(hidden_terms))[:5]
        if not terms:
            continue
        source_key = (hit.get("source_path", hit["filename"]), hit["chunk_index"], tuple(terms))
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)
        bullet = build_cv_bullet(hit, terms)
        verification = verify_rewrite_claim(hit, terms, bullet)
        suggestions.append(
            {
                "bullet": bullet,
                "target_terms": terms,
                "source_category": hit.get("category", "general"),
                "source_path": hit.get("source_path", hit["filename"]),
                "chunk_index": hit["chunk_index"],
                "evidence_excerpt": trim_sentence(re.sub(r"\s+", " ", hit["text"]).strip(), 260),
                "confidence": rewrite_confidence(hit, terms),
                "safe_to_claim": verification["status"] == "supported",
                "claim_verification": verification,
            }
        )
        if len(suggestions) == 6:
            break
    return suggestions


def build_cv_bullet(hit: dict, terms: list[str]) -> str:
    category = hit.get("category", "general")
    verb = "Delivered" if category == "experience" else "Built"
    readable_terms = join_terms(terms)
    summary = re.sub(r"\s+", " ", hit["text"]).strip()
    evidence = trim_sentence(summary, 150)
    return (
        f"{verb} work demonstrating {readable_terms}, grounded in: {evidence}. "
        "Quantify impact if true."
    )


def rewrite_confidence(hit: dict, terms: list[str]) -> float:
    score = float(hit.get("score", 0.0))
    term_bonus = min(0.25, len(terms) * 0.05)
    return round(min(0.95, max(0.45, score + term_bonus)), 2)


def verify_rewrite_claim(hit: dict, terms: list[str], bullet: str) -> dict:
    text = re.sub(r"\s+", " ", hit.get("text", "")).casefold()
    bullet_text = bullet.casefold()
    supported_terms = [term for term in terms if term in skill_terms(text)]
    risky_claims = []
    required_evidence = []
    risk_markers = {
        "production": "production deployment evidence",
        "deployed": "deployment evidence",
        "users": "user or usage evidence",
        "%": "measured impact",
        "improved": "before/after metric",
        "reduced": "before/after metric",
        "increased": "before/after metric",
    }
    for marker, evidence in risk_markers.items():
        if marker in bullet_text and marker not in text:
            risky_claims.append(marker)
            required_evidence.append(evidence)
    if not supported_terms:
        return {
            "status": "unsupported",
            "reason": "The suggested bullet does not have direct term support in the retrieved evidence.",
            "supported_claims": [],
            "risky_claims": sorted(set(risky_claims)),
            "required_evidence": sorted(set(required_evidence or ["direct source evidence"])),
        }
    if risky_claims:
        return {
            "status": "needs_verification",
            "reason": "Core skills are supported, but impact or deployment language needs explicit evidence before claiming it.",
            "supported_claims": supported_terms,
            "risky_claims": sorted(set(risky_claims)),
            "required_evidence": sorted(set(required_evidence)),
        }
    return {
        "status": "supported",
        "reason": "The target terms appear in the cited project or experience evidence.",
        "supported_claims": supported_terms,
        "risky_claims": [],
        "required_evidence": [],
    }


def build_score_explanations(
    *,
    requirements: dict,
    scoring_breakdown: dict,
    evidence_depth: dict,
    resume_hits: list[dict],
    matched: list[str],
    weak_evidence: list[str],
) -> list[dict]:
    by_term = evidence_depth.get("by_term", {})
    required = requirements.get("required", [])
    preferred = requirements.get("preferred", [])
    responsibilities = requirements.get("responsibilities", [])
    credible_required = [
        term
        for term in required
        if by_term.get(term, {}).get("cv_depth") in {"work_or_internship", "production_or_measurable_impact"}
    ]
    credible_preferred = [
        term
        for term in preferred
        if by_term.get(term, {}).get("cv_depth") in {"work_or_internship", "production_or_measurable_impact"}
    ]
    covered_responsibilities = [
        term
        for term in responsibilities
        if by_term.get(term, {}).get("cv_depth") in {"work_or_internship", "production_or_measurable_impact"}
    ]
    impact_evidence = quantified_impact_evidence(resume_hits)
    return [
        {
            "component": "required_skill_coverage",
            "points": scoring_breakdown.get("required_skill_coverage", 0),
            "max_points": 45,
            "evidence": credible_required[:10],
            "missing": [term for term in required if term not in credible_required][:10],
            "reason": "Required skills earn full credit only when the current CV has credible evidence, not just project-only evidence.",
        },
        {
            "component": "responsibility_alignment",
            "points": scoring_breakdown.get("responsibility_alignment", 0),
            "max_points": 20,
            "evidence": covered_responsibilities[:10],
            "missing": [term for term in responsibilities if term not in covered_responsibilities][:10],
            "reason": "Responsibility alignment checks whether the CV demonstrates the work patterns implied by the JD.",
        },
        {
            "component": "evidence_depth",
            "points": scoring_breakdown.get("evidence_depth", 0),
            "max_points": 20,
            "evidence": matched[:10],
            "missing": weak_evidence[:10],
            "reason": "Shallow mentions score lower than internship, work, production, or measured-impact evidence.",
        },
        {
            "component": "quantified_impact",
            "points": scoring_breakdown.get("quantified_impact", 0),
            "max_points": 10,
            "evidence": impact_evidence[:3],
            "missing": [] if impact_evidence else ["measured impact in the current CV"],
            "reason": "Impact credit requires a number such as percent improvement, users, rows, requests, time, or similar measurable output.",
        },
        {
            "component": "preferred_coverage",
            "points": scoring_breakdown.get("preferred_coverage", 0),
            "max_points": 5,
            "evidence": credible_preferred[:10],
            "missing": [term for term in preferred if term not in credible_preferred][:10],
            "reason": "Preferred requirements can improve the score, but they do not compensate for missing required evidence.",
        },
    ]


def quantified_impact_evidence(hits: list[dict]) -> list[str]:
    evidence = []
    for hit in hits:
        text = re.sub(r"\s+", " ", hit.get("text", "")).strip()
        if re.search(r"(\d+%|\$\d+|\d+\s*(users|requests|records|rows|hours|seconds|x|ms)\b)", text.casefold()):
            evidence.append(trim_sentence(text, 180))
    return evidence


def join_terms(terms: list[str]) -> str:
    if not terms:
        return "the target requirement"
    if len(terms) == 1:
        return terms[0]
    return ", ".join(terms[:-1]) + f", and {terms[-1]}"


def trim_sentence(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    trimmed = text[:limit].rsplit(" ", 1)[0]
    return trimmed + "..."


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
    application_verdict: dict,
    cv_rewrite_suggestions: list[dict],
    requirements: dict,
    evidence_depth: dict,
    scoring_breakdown: dict,
    weak_evidence: list[str],
    matched: list[str],
    hidden_terms: list[str],
    gaps: list[str],
    resume_hits: list[dict],
    supporting_hits: list[dict],
) -> str | None:
    resume_context = format_hits_for_prompt(resume_hits[:4])
    supporting_context = format_hits_for_prompt(supporting_hits[:5])
    rewrite_context = "\n".join(
        f"- {item['bullet']} (source: {item['source_path']} chunk {item['chunk_index']})"
        for item in cv_rewrite_suggestions
    )
    return nvidia_chat(
        system_prompt=(
            "You are a brutally honest career reviewer for data scientist and AI roles. "
            "Use only the supplied JD, CV evidence, and project/experience evidence. "
            "Do not flatter the candidate. If the evidence is insufficient, say so clearly. "
            "Separate current CV fit from recommendations to improve the CV. "
            "Do not invent metrics, employers, outcomes, or tools."
        ),
        user_prompt=(
            f"Job title: {job_title}\n"
            f"JD excerpt:\n{job_text[:3500]}\n\n"
            f"Deterministic JD-to-CV score: {fit_score}/100\n"
            f"Deterministic application verdict: {application_verdict}\n"
            f"Role family: {requirements.get('role_family')}\n"
            f"JD requirements: {requirements}\n"
            f"Scoring breakdown: {scoring_breakdown}\n"
            f"Weak evidence: {', '.join(weak_evidence) or 'none'}\n"
            f"Evidence depth: {evidence_depth}\n"
            f"CV matched terms: {', '.join(matched) or 'none'}\n"
            f"Terms found in projects/experience but missing from CV: {', '.join(hidden_terms) or 'none'}\n"
            f"Terms with no evidence: {', '.join(gaps) or 'none'}\n\n"
            f"CV evidence:\n{resume_context or 'No CV evidence.'}\n\n"
            f"Project/experience evidence:\n{supporting_context or 'No project/experience evidence.'}\n\n"
            f"Grounded draft CV bullets:\n{rewrite_context or 'No grounded rewrite suggestions.'}\n\n"
            "Return concise feedback with: verdict, why, improved CV bullets, and whether applying is realistic."
        ),
    )


def format_hits_for_prompt(hits: list[dict]) -> str:
    lines = []
    for index, hit in enumerate(hits, start=1):
        text = re.sub(r"\s+", " ", hit["text"]).strip()[:700]
        lines.append(f"[{index}] {hit.get('category', 'general')}/{hit['filename']} chunk {hit['chunk_index']}: {text}")
    return "\n".join(lines)
