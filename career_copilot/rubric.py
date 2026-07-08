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
REQUIRED_SECTION_MARKERS = {
    "required",
    "required qualifications",
    "minimum qualifications",
    "basic qualifications",
    "must have",
    "requirements",
    "qualifications",
}
PREFERRED_SECTION_MARKERS = {
    "preferred",
    "preferred qualifications",
    "nice to have",
    "bonus",
    "plus",
}
RESPONSIBILITY_SECTION_MARKERS = {
    "responsibilities",
    "what you will do",
    "what you'll do",
    "role responsibilities",
    "day to day",
}
CONTEXT_SECTION_MARKERS = {
    "about us",
    "about the team",
    "benefits",
    "company",
    "why join",
}

ROLE_TERM_WEIGHTS = {
    "ai_engineer": {
        "rag": 1.3,
        "llm": 1.25,
        "retrieval": 1.2,
        "evaluation": 1.15,
        "deployment": 1.1,
        "api": 1.05,
        "python": 1.0,
        "embedding": 0.75,
        "vector_search": 0.9,
        "prompting": 0.45,
        "dashboard": 0.25,
    },
    "data_scientist": {
        "sql": 1.25,
        "statistics": 1.25,
        "ab_test": 1.2,
        "forecasting": 1.15,
        "dashboard": 1.1,
        "pandas": 1.0,
        "machine_learning": 0.9,
        "python": 0.9,
        "rag": 0.35,
        "llm": 0.4,
        "embedding": 0.3,
        "prompting": 0.25,
    },
    "ml_engineer": {
        "machine_learning": 1.25,
        "deployment": 1.25,
        "mlops": 1.2,
        "docker": 1.05,
        "kubernetes": 1.1,
        "api": 1.0,
        "python": 1.0,
        "evaluation": 1.0,
        "dashboard": 0.3,
        "prompting": 0.35,
    },
    "general_ai_data": {},
}
SECTION_WEIGHT = {
    "required": 1.0,
    "responsibility": 0.85,
    "preferred": 0.55,
    "context": 0.2,
}
MIN_REQUIRED_WEIGHT = 0.5


def analyze_job_requirements(job_text: str) -> dict:
    sections = split_job_sections(job_text)
    text = normalize(job_text)
    role_family = detect_role_family(text)
    term_details = build_term_details(sections, role_family)
    preferred = sorted(term for term, detail in term_details.items() if detail["strength"] == "preferred")
    context = sorted(term for term, detail in term_details.items() if detail["strength"] == "context")
    required = sorted(
        term
        for term, detail in term_details.items()
        if detail["strength"] in {"required", "responsibility"} and detail["weight"] >= MIN_REQUIRED_WEIGHT
    )
    responsibilities = sorted(extract_responsibilities_by_section(sections))
    ignored = sorted(term for term, detail in term_details.items() if detail["weight"] < MIN_REQUIRED_WEIGHT and detail["strength"] != "preferred")
    return {
        "role_family": role_family,
        "required": required,
        "preferred": preferred,
        "context": context,
        "ignored": ignored,
        "responsibilities": responsibilities,
        "term_details": [term_details[term] for term in sorted(term_details)],
        "term_weights": {term: detail["weight"] for term, detail in term_details.items()},
        "all_terms": sorted(set(required) | set(preferred) | set(context) | set(responsibilities)),
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
    term_weights = requirements.get("term_weights") or {}
    by_term = depth.get("by_term", {})

    mentioned_required = covered_terms(required, by_term)
    mentioned_preferred = covered_terms(preferred, by_term)
    required_covered = credible_terms(required, by_term)
    preferred_covered = credible_terms(preferred, by_term)
    responsibility_covered = credible_terms(responsibilities, by_term)

    required_component = 45 * weighted_ratio(required_covered, required, term_weights)
    responsibility_component = 20 * weighted_ratio(responsibility_covered, responsibilities or required, term_weights)
    depth_component = 20 * weighted_average_depth(required, by_term, term_weights)
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


def split_job_sections(job_text: str) -> list[dict]:
    sections: list[dict] = []
    current = {"section": "required", "heading": "unheaded", "text": ""}
    for raw_line in job_text.splitlines():
        line = raw_line.strip(" -\t")
        if not line:
            continue
        heading_part, inline_text = split_heading_line(line)
        heading = normalize(heading_part.strip("#:"))
        section = classify_heading(heading)
        if section and len(heading_part) <= 80:
            if current["text"].strip():
                sections.append(current)
            current = {"section": section, "heading": heading, "text": inline_text}
            continue
        current["text"] += " " + line
    if current["text"].strip():
        sections.append(current)
    if not sections:
        sections.append({"section": "required", "heading": "unheaded", "text": job_text})
    return sections


def split_heading_line(line: str) -> tuple[str, str]:
    if ":" not in line:
        return line, ""
    heading, rest = line.split(":", 1)
    return heading, rest.strip()


def classify_heading(heading: str) -> str | None:
    if any(marker == heading or marker in heading for marker in PREFERRED_SECTION_MARKERS):
        return "preferred"
    if any(marker == heading or marker in heading for marker in REQUIRED_SECTION_MARKERS):
        return "required"
    if any(marker == heading or marker in heading for marker in RESPONSIBILITY_SECTION_MARKERS):
        return "responsibility"
    if any(marker == heading or marker in heading for marker in CONTEXT_SECTION_MARKERS):
        return "context"
    return None


def build_term_details(sections: list[dict], role_family: str) -> dict[str, dict]:
    details: dict[str, dict] = {}
    for section in sections:
        section_type = section.get("section", "context")
        text = normalize(section.get("text", ""))
        section_skills = extract_skills(text)
        preferred_skills = extract_preferred_skills(text, section_skills)
        for term in section_skills:
            inferred_section = "preferred" if term in preferred_skills or section_type == "preferred" else section_type
            if inferred_section == "context" and has_required_marker_near_term(text, term):
                inferred_section = "required"
            weight = term_weight(term, role_family, inferred_section)
            existing = details.get(term)
            candidate = {
                "term": term,
                "strength": inferred_section,
                "section": section_type,
                "weight": weight,
                "role_weight": role_term_weight(term, role_family),
                "section_weight": SECTION_WEIGHT.get(inferred_section, 0.2),
                "reason": build_term_reason(term, role_family, inferred_section, weight),
            }
            if existing is None or candidate["weight"] > existing["weight"]:
                details[term] = candidate
    return details


def term_weight(term: str, role_family: str, section: str) -> float:
    return round(role_term_weight(term, role_family) * SECTION_WEIGHT.get(section, 0.2), 3)


def role_term_weight(term: str, role_family: str) -> float:
    return ROLE_TERM_WEIGHTS.get(role_family, {}).get(term, 0.85)


def build_term_reason(term: str, role_family: str, section: str, weight: float) -> str:
    if section == "preferred":
        return f"{term} appears as preferred or nice-to-have evidence for {role_family}."
    if section == "context":
        return f"{term} appears in contextual JD text and is down-weighted for {role_family}."
    if weight < MIN_REQUIRED_WEIGHT:
        return f"{term} is low-priority for {role_family} and is not treated as a core requirement."
    return f"{term} is treated as {section} evidence for {role_family}."


def has_required_marker_near_term(text: str, term: str) -> bool:
    aliases = SKILL_ALIASES.get(term, {term})
    sentences = split_sentences(text)
    for sentence in sentences:
        if any(alias in sentence for alias in aliases) and any(marker in sentence for marker in {"required", "must", "need", "experience with"}):
            return True
    return False


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


def extract_responsibilities_by_section(sections: list[dict]) -> set[str]:
    found = set()
    for section in sections:
        section_type = section.get("section", "context")
        text = normalize(section.get("text", ""))
        section_responsibilities = extract_responsibilities(text)
        if section_type in {"responsibility", "required"}:
            found |= section_responsibilities
        else:
            for responsibility in section_responsibilities:
                if role_action_is_explicit(text, responsibility):
                    found.add(responsibility)
    return found


def role_action_is_explicit(text: str, responsibility: str) -> bool:
    aliases = RESPONSIBILITY_ALIASES.get(responsibility, {responsibility})
    return any(alias in text for alias in aliases) and any(marker in text for marker in {"you will", "responsible", "build", "develop", "deploy", "analyze", "communicate"})


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


def weighted_ratio(covered: Iterable[str], terms: Iterable[str], weights: dict[str, float]) -> float:
    term_list = list(terms)
    if not term_list:
        return 0.0
    covered_set = set(covered)
    total = sum(term_score_weight(term, weights) for term in term_list)
    earned = sum(term_score_weight(term, weights) for term in term_list if term in covered_set)
    return earned / total if total else 0.0


def weighted_average_depth(terms: list[str], by_term: dict, weights: dict[str, float]) -> float:
    if not terms:
        return 0.0
    total = sum(term_score_weight(term, weights) for term in terms)
    earned = sum(DEPTH_POINTS[cv_depth(term, by_term)] * term_score_weight(term, weights) for term in terms)
    return earned / total if total else 0.0


def term_score_weight(term: str, weights: dict[str, float]) -> float:
    return max(0.1, float(weights.get(term, 1.0)))


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
