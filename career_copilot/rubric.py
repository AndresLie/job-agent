from __future__ import annotations

import re
from collections.abc import Iterable

from .vector_store import keyword_tokens


DEPTH_ORDER = {
    "none": 0,
    "mentioned": 1,
    "course_or_class_project": 2,
    "portfolio_project": 3,
    "work_or_internship": 4,
    "production_or_measurable_impact": 5,
}

DEPTH_POINTS = {
    "none": 0.0,
    "mentioned": 0.2,
    "course_or_class_project": 0.35,
    "portfolio_project": 0.5,
    "work_or_internship": 0.75,
    "production_or_measurable_impact": 1.0,
}

SKILL_ALIASES = {
    "ab_test": {"a/b", "ab test", "a b test", "experiment", "experimentation"},
    "agent": {"agent", "agents", "agentic"},
    "airflow": {"airflow"},
    "api": {"api", "apis", "fastapi", "flask"},
    "aws": {"aws", "s3", "lambda", "ec2"},
    "causal": {"causal", "causality", "causal inference"},
    "dashboard": {"dashboard", "dashboards", "bi", "tableau", "power bi", "looker"},
    "deep_learning": {"deep learning", "neural", "transformer", "cnn", "rnn"},
    "deployment": {"deploy", "deployed", "deployment", "serving", "inference"},
    "docker": {"docker", "container", "containers"},
    "embedding": {"embedding", "embeddings"},
    "evaluation": {"evaluation", "evaluate", "eval", "benchmark", "metrics"},
    "forecasting": {"forecast", "forecasting", "time series"},
    "guardrails": {"guardrail", "guardrails", "safety"},
    "kubernetes": {"kubernetes", "k8s"},
    "llm": {"llm", "llms", "large language model", "large language models", "gemma", "gpt"},
    "machine_learning": {"machine learning", "ml", "modeling", "model", "models"},
    "mlops": {"mlops", "ci cd", "monitoring", "model registry"},
    "pandas": {"pandas", "dataframe", "dataframes"},
    "prompting": {"prompt", "prompting", "prompt engineering"},
    "python": {"python"},
    "pytorch": {"pytorch", "torch"},
    "rag": {"rag", "retrieval augmented generation"},
    "retrieval": {"retrieval", "retrieve", "search"},
    "sklearn": {"sklearn", "scikit", "scikit learn"},
    "spark": {"spark", "pyspark"},
    "sql": {"sql", "postgres", "postgresql", "mysql", "sqlite", "bigquery"},
    "statistics": {"statistics", "statistical", "stats", "probability"},
    "tensorflow": {"tensorflow", "keras"},
    "vector_search": {"vector search", "vector database", "vector store", "faiss", "pinecone", "qdrant"},
}

ROLE_HINTS = {
    "data_scientist": {
        "data scientist",
        "statistics",
        "experiment",
        "experimentation",
        "a/b",
        "dashboard",
        "analytics",
        "forecast",
        "business metrics",
    },
    "ai_engineer": {"ai engineer", "genai", "generative ai", "rag", "llm", "agent", "prompt"},
    "ml_engineer": {"ml engineer", "machine learning engineer", "deployment", "mlops", "serving", "kubernetes"},
}

RESPONSIBILITY_ALIASES = {
    "build_models": {"build model", "develop model", "train model", "modeling", "machine learning"},
    "evaluate_systems": {"evaluate", "evaluation", "benchmark", "experiment", "metrics"},
    "deploy_systems": {"deploy", "deployment", "production", "serving", "api"},
    "analyze_data": {"analyze", "analysis", "insight", "analytics", "dashboard"},
    "communicate_results": {"communicate", "stakeholder", "present", "report"},
}

PREFERRED_MARKERS = {"preferred", "nice to have", "bonus", "plus", "familiarity"}


def analyze_job_requirements(job_text: str) -> dict:
    text = normalize(job_text)
    role_family = detect_role_family(text)
    skills = sorted(extract_skills(text))
    preferred = sorted(extract_preferred_skills(text, skills))
    required = [skill for skill in skills if skill not in preferred]
    responsibilities = sorted(extract_responsibilities(text))
    return {
        "role_family": role_family,
        "required": required,
        "preferred": preferred,
        "responsibilities": responsibilities,
        "all_terms": sorted(set(required) | set(preferred) | set(responsibilities)),
    }


def analyze_evidence_depth(requirements: dict, resume_hits: list[dict], supporting_hits: list[dict]) -> dict:
    terms = requirements.get("all_terms") or []
    by_term = {}
    for term in terms:
        resume_depth = best_depth_for_term(term, resume_hits)
        supporting_depth = best_depth_for_term(term, supporting_hits)
        by_term[term] = {
            "cv_depth": resume_depth,
            "supporting_depth": supporting_depth,
            "best_depth": max_depth(resume_depth, supporting_depth),
        }
    return {
        "by_term": by_term,
        "weak_evidence": sorted(
            term
            for term in requirements.get("required", [])
            if by_term.get(term, {}).get("cv_depth") in {"none", "mentioned", "course_or_class_project"}
        ),
        "strong_evidence": sorted(
            term
            for term, data in by_term.items()
            if DEPTH_ORDER.get(data.get("cv_depth", "none"), 0) >= DEPTH_ORDER["work_or_internship"]
        ),
    }


def score_with_rubric(requirements: dict, depth: dict, resume_hits: list[dict]) -> dict:
    required = requirements.get("required") or requirements.get("all_terms") or []
    preferred = requirements.get("preferred") or []
    responsibilities = requirements.get("responsibilities") or []
    by_term = depth.get("by_term", {})

    mentioned_required = covered_terms(required, by_term)
    mentioned_preferred = covered_terms(preferred, by_term)
    required_covered = credible_terms(required, by_term)
    preferred_covered = credible_terms(preferred, by_term)
    responsibility_covered = credible_terms(responsibilities, by_term)

    required_component = 45 * ratio(len(required_covered), len(required))
    responsibility_component = 20 * ratio(len(responsibility_covered), len(responsibilities) or len(required))
    depth_component = 20 * average_depth(required, by_term)
    impact_component = 10 * quantified_impact_score(resume_hits)
    preferred_component = 5 * ratio(len(preferred_covered), len(preferred))
    raw_score = round(
        required_component
        + responsibility_component
        + depth_component
        + impact_component
        + preferred_component
    )
    fit_score = min(100, max(0, raw_score))

    return {
        "fit_score": fit_score,
        "matched_terms": sorted(mentioned_required | mentioned_preferred),
        "credible_matched_terms": sorted(required_covered | preferred_covered),
        "missing_from_cv": sorted(term for term in set(required) | set(preferred) if cv_depth(term, by_term) == "none"),
        "hidden_terms": sorted(
            term
            for term in set(required) | set(preferred)
            if cv_depth(term, by_term) == "none" and supporting_depth(term, by_term) != "none"
        ),
        "skill_gaps": sorted(
            term
            for term in set(required) | set(preferred)
            if cv_depth(term, by_term) == "none" and supporting_depth(term, by_term) == "none"
        ),
        "weak_evidence": depth.get("weak_evidence", []),
        "scoring_breakdown": {
            "required_skill_coverage": round(required_component, 2),
            "responsibility_alignment": round(responsibility_component, 2),
            "evidence_depth": round(depth_component, 2),
            "quantified_impact": round(impact_component, 2),
            "preferred_coverage": round(preferred_component, 2),
            "credible_required_matches": len(required_covered),
            "mentioned_required_matches": len(mentioned_required),
        },
    }


def known_skill_terms() -> set[str]:
    return set(SKILL_ALIASES)


def detect_role_family(text: str) -> str:
    scores = {
        role: sum(1 for hint in hints if hint in text)
        for role, hints in ROLE_HINTS.items()
    }
    best_role, best_score = max(scores.items(), key=lambda item: item[1])
    return best_role if best_score else "general_ai_data"


def extract_skills(text: str) -> set[str]:
    found = set()
    token_set = keyword_tokens(text)
    for skill, aliases in SKILL_ALIASES.items():
        if skill in token_set or any(alias in text for alias in aliases):
            found.add(skill)
    return found


def extract_preferred_skills(text: str, skills: Iterable[str]) -> set[str]:
    preferred = set()
    sentences = split_sentences(text)
    for sentence in sentences:
        if any(marker in sentence for marker in PREFERRED_MARKERS):
            preferred |= extract_skills(sentence)
    return preferred & set(skills)


def extract_responsibilities(text: str) -> set[str]:
    found = set()
    for responsibility, aliases in RESPONSIBILITY_ALIASES.items():
        if any(alias in text for alias in aliases):
            found.add(responsibility)
    return found


def best_depth_for_term(term: str, hits: list[dict]) -> str:
    best = "none"
    for hit in hits:
        text = normalize(hit.get("text", ""))
        if not term_in_text(term, text):
            continue
        depth = classify_depth(text, hit.get("category", "general"))
        if DEPTH_ORDER[depth] > DEPTH_ORDER[best]:
            best = depth
    return best


def classify_depth(text: str, category: str) -> str:
    if has_quantified_impact(text) or "production" in text or "users" in text or "deployed" in text:
        return "production_or_measurable_impact"
    if category == "experience" or any(word in text for word in {"internship", "intern", "work", "worked", "company"}):
        return "work_or_internship"
    if category == "projects":
        if any(word in text for word in {"course", "class", "homework", "assignment"}):
            return "course_or_class_project"
        return "portfolio_project"
    return "mentioned"


def has_quantified_impact(text: str) -> bool:
    return bool(re.search(r"(\d+%|\$\d+|\d+\s*(users|requests|records|rows|hours|seconds|x|ms)\b)", text))


def term_in_text(term: str, text: str) -> bool:
    aliases = SKILL_ALIASES.get(term) or RESPONSIBILITY_ALIASES.get(term) or {term}
    return term in keyword_tokens(text) or any(alias in text for alias in aliases)


def covered_terms(terms: Iterable[str], by_term: dict) -> set[str]:
    return {term for term in terms if cv_depth(term, by_term) != "none"}


def credible_terms(terms: Iterable[str], by_term: dict) -> set[str]:
    return {
        term
        for term in terms
        if DEPTH_ORDER.get(cv_depth(term, by_term), 0) >= DEPTH_ORDER["work_or_internship"]
    }


def average_depth(terms: list[str], by_term: dict) -> float:
    if not terms:
        return 0.0
    return sum(DEPTH_POINTS[cv_depth(term, by_term)] for term in terms) / len(terms)


def quantified_impact_score(resume_hits: list[dict]) -> float:
    return 1.0 if any(has_quantified_impact(normalize(hit.get("text", ""))) for hit in resume_hits) else 0.0


def cv_depth(term: str, by_term: dict) -> str:
    return by_term.get(term, {}).get("cv_depth", "none")


def supporting_depth(term: str, by_term: dict) -> str:
    return by_term.get(term, {}).get("supporting_depth", "none")


def max_depth(left: str, right: str) -> str:
    return left if DEPTH_ORDER[left] >= DEPTH_ORDER[right] else right


def ratio(left: int, right: int) -> float:
    if right <= 0:
        return 0.0
    return left / right


def split_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"[.\n;:]", text) if sentence.strip()]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()
