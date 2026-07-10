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
    "ab_test": {"a/b", "ab test", "a b test", "experiment", "experiments", "experimentation"},
    "agent": {"agent", "agents", "agentic"},
    "airflow": {"airflow"},
    "api": {"api", "apis", "fastapi", "flask"},
    "aws": {"aws", "s3", "lambda", "ec2"},
    "causal": {"causal", "causality", "causal inference"},
    "csharp": {"c#", "c sharp", "csharp"},
    "cpp": {"c++", "cpp"},
    "dashboard": {"dashboard", "dashboards", "bi", "tableau", "power bi", "looker"},
    "deep_learning": {"deep learning", "neural", "transformer", "cnn", "rnn"},
    "deployment": {"deploy", "deployed", "deployment", "serving", "inference"},
    "docker": {"docker", "container", "containers"},
    "embedding": {"embedding", "embeddings"},
    "evaluation": {"evaluation", "evaluate", "eval", "benchmark", "metrics"},
    "forecasting": {"forecast", "forecasting", "time series"},
    "guardrails": {"guardrail", "guardrails", "safety"},
    "incident_management": {"incident", "incident management", "problem management", "itsm", "servicenow"},
    "javascript": {"javascript", "java script", "js"},
    "git": {"git", "gitlab"},
    "kubernetes": {"kubernetes", "k8s"},
    "linux": {"linux"},
    "llm": {"llm", "llms", "large language model", "large language models", "gemma", "gpt"},
    "machine_learning": {"machine learning", "ml", "modeling", "model", "models"},
    "mes": {"mes", "manufacturing execution system", "manufacturing execution systems"},
    "monitoring": {"monitoring", "alerting", "logging", "splunk"},
    "mlops": {"mlops", "ci cd", "monitoring", "model registry"},
    "nodejs": {"nodejs", "node.js", "node js"},
    "nosql": {"nosql", "no sql", "mongodb", "mongo", "cassandra", "dynamodb", "redis", "iotdb"},
    "oop": {"object oriented", "object-oriented", "oop"},
    "openshift": {"openshift", "open shift"},
    "oracle": {"oracle"},
    "pandas": {"pandas", "dataframe", "dataframes"},
    "perl": {"perl"},
    "pl_sql": {"pl/sql", "pl sql"},
    "prompting": {"prompt", "prompting", "prompt engineering"},
    "production_support": {"production support", "technical support", "support software", "support software systems", "on-call", "on call"},
    "python": {"python"},
    "pytorch": {"pytorch", "torch"},
    "rag": {"rag", "retrieval augmented generation"},
    "release_testing": {"release testing", "testing of new software releases", "software releases"},
    "retrieval": {"retrieval", "retrieve", "search"},
    "sklearn": {"sklearn", "scikit", "scikit learn"},
    "spark": {"spark", "pyspark"},
    "sql": {"sql", "microsoft sql", "ms sql", "postgres", "postgresql", "mysql", "sqlite", "bigquery"},
    "statistics": {"statistics", "statistical", "stats", "probability"},
    "tensorflow": {"tensorflow", "keras"},
    "troubleshooting": {"troubleshoot", "troubleshooting", "debugging", "problem identification", "problem resolution"},
    "typescript": {"typescript", "type script", "ts"},
    "unix": {"unix"},
    "vector_search": {"vector search", "vector database", "vector store", "faiss", "pinecone", "qdrant"},
    "windows": {"windows", "windows server"},
    "angular": {"angular", "angularjs", "angular.js"},
    "apache": {"apache"},
    "code_generation_tools": {
        "github copilot",
        "copilot",
        "claude code",
        "cursor",
        "tabnine",
        "code generation",
        "code-generation",
    },
    "dotnet": {".net", "dotnet", "asp.net"},
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
    "ai_engineer": {"ai engineer", "generative ai engineer", "rag", "llm", "agent", "prompt"},
    "ml_engineer": {"ml engineer", "machine learning engineer", "deployment", "mlops", "serving", "kubernetes"},
    "manufacturing_it": {
        "manufacturing it",
        "mes",
        "manufacturing execution",
        "semiconductor manufacturing",
        "photolithography",
        "fabs",
        "fab",
        "smart manufacturing",
        "smart mfg",
        "incident management",
        "problem management",
        "production support",
        "technical support",
        "microsoft sql",
        "oracle",
        "c#",
        "c++",
        "perl",
    },
    "software_engineer": {
        "software engineer",
        "software development engineer",
        "full-stack",
        "full stack",
        "software systems",
        "software applications",
        "programming",
        "scripting",
        "database technologies",
        "restful web services",
        "web technologies",
        "nodejs",
        "node.js",
        ".net",
        "angular",
        "typescript",
        "linux",
        "windows",
    },
}

RESPONSIBILITY_ALIASES = {
    "build_models": {"build model", "develop model", "train model", "modeling", "machine learning"},
    "evaluate_systems": {"evaluate", "evaluation", "benchmark", "experiment", "metrics"},
    "deploy_systems": {"deploy", "deployment", "production", "serving", "api"},
    "analyze_data": {"analyze", "analysis", "insight", "analytics", "dashboard"},
    "communicate_results": {"communicate", "stakeholder", "present", "report"},
    "support_systems": {"support", "technical support", "production support", "incident", "problem management", "troubleshoot"},
    "test_releases": {"testing of new software releases", "software releases", "release testing"},
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
    "manufacturing_it": {
        "mes": 1.35,
        "sql": 1.2,
        "oracle": 1.1,
        "pl_sql": 1.05,
        "csharp": 1.15,
        "cpp": 1.15,
        "perl": 1.0,
        "linux": 1.0,
        "unix": 1.0,
        "windows": 1.0,
        "incident_management": 1.2,
        "production_support": 1.25,
        "troubleshooting": 1.2,
        "release_testing": 0.95,
        "monitoring": 0.95,
        "deployment": 0.95,
        "api": 0.85,
        "python": 0.75,
        "rag": 0.2,
        "llm": 0.2,
        "aws": 0.35,
        "dashboard": 0.45,
    },
    "software_engineer": {
        "api": 1.1,
        "deployment": 1.0,
        "sql": 1.0,
        "nosql": 0.95,
        "csharp": 1.0,
        "cpp": 1.0,
        "nodejs": 1.05,
        "apache": 0.9,
        "dotnet": 1.05,
        "angular": 1.0,
        "typescript": 1.0,
        "javascript": 0.95,
        "git": 0.9,
        "oop": 0.95,
        "docker": 0.9,
        "kubernetes": 0.9,
        "openshift": 0.85,
        "code_generation_tools": 0.55,
        "python": 0.95,
        "linux": 0.9,
        "windows": 0.85,
        "troubleshooting": 0.95,
        "production_support": 0.9,
        "rag": 0.35,
        "dashboard": 0.45,
    },
    "general_ai_data": {},
}
CORE_STACK_TERMS = {
    "nodejs",
    "apache",
    "csharp",
    "dotnet",
    "angular",
    "typescript",
    "javascript",
    "api",
    "sql",
    "nosql",
    "git",
    "oop",
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
    role_profile = detect_role_profile(text)
    role_family = role_profile["primary"]
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
    requirement_groups = build_requirement_groups(sections, term_details)
    return {
        "role_family": role_family,
        "role_profile": role_profile,
        "required": required,
        "preferred": preferred,
        "context": context,
        "ignored": ignored,
        "responsibilities": responsibilities,
        "requirement_groups": requirement_groups,
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

    required_units = requirement_units(requirements, "required")
    required_component = 45 * weighted_unit_ratio(required_covered, required_units, term_weights)
    responsibility_component = 20 * weighted_ratio(responsibility_covered, responsibilities or required, term_weights)
    depth_component = 20 * weighted_average_unit_depth(required_units, by_term, term_weights)
    impact_component = 10 * quantified_impact_score(resume_hits)
    preferred_component = 5 * ratio(len(preferred_covered), len(preferred))
    raw_score = round(
        required_component
        + responsibility_component
        + depth_component
        + impact_component
        + preferred_component
    )
    cap = score_cap(requirements, by_term)
    fit_score = min(100, max(0, raw_score))
    if cap["cap"] is not None:
        fit_score = min(fit_score, cap["cap"])

    return {
        "fit_score": fit_score,
        "matched_terms": sorted(mentioned_required | mentioned_preferred),
        "credible_matched_terms": sorted(required_covered | preferred_covered),
        "missing_from_cv": missing_terms_for_units(requirements, by_term, include_preferred=True),
        "hidden_terms": sorted(
            term
            for term in set(missing_terms_for_units(requirements, by_term, include_preferred=True))
            if cv_depth(term, by_term) == "none" and supporting_depth(term, by_term) != "none"
        ),
        "skill_gaps": skill_gap_terms(requirements, by_term),
        "weak_evidence": depth.get("weak_evidence", []),
        "scoring_breakdown": {
            "required_skill_coverage": round(required_component, 2),
            "responsibility_alignment": round(responsibility_component, 2),
            "evidence_depth": round(depth_component, 2),
            "quantified_impact": round(impact_component, 2),
            "preferred_coverage": round(preferred_component, 2),
            "credible_required_matches": len(required_covered),
            "mentioned_required_matches": len(mentioned_required),
            "raw_score_before_caps": raw_score,
            "score_cap": cap["cap"],
            "score_cap_reason": cap["reason"],
            "missing_core_stack_groups": cap["missing_groups"],
        },
    }


def known_skill_terms() -> set[str]:
    return set(SKILL_ALIASES)


def detect_role_family(text: str) -> str:
    return detect_role_profile(text)["primary"]


def detect_role_profile(text: str) -> dict:
    normalized_text = normalize(text)
    scores = score_role_families(normalized_text)
    primary, best_score = max(scores.items(), key=lambda item: item[1])
    primary = primary if best_score else "general_ai_data"
    secondary = [
        role
        for role, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if role != primary and score >= 2.0
    ][:2]
    if role_has_strong_software_signal(normalized_text):
        primary = "software_engineer"
        secondary = [role for role in secondary if role != primary]
        if scores.get("manufacturing_it", 0) >= 2.0 and "manufacturing_it" not in secondary:
            secondary.insert(0, "manufacturing_it")
        if scores.get("ai_engineer", 0) >= 2.0 and "ai_engineer" not in secondary:
            secondary.append("ai_engineer")
    if (
        primary == "software_engineer"
        and role_has_manufacturing_system_signal(normalized_text)
        and not role_has_full_stack_product_signal(normalized_text)
    ):
        primary = "manufacturing_it"
        secondary = ["software_engineer"] + [
            role for role in secondary if role not in {"software_engineer", "manufacturing_it"}
        ]
    if primary == "manufacturing_it" and scores.get("software_engineer", 0) >= scores.get("manufacturing_it", 0) - 1.0:
        if role_has_full_stack_stack_signal(normalized_text):
            primary = "software_engineer"
            secondary = ["manufacturing_it"] + [role for role in secondary if role not in {"software_engineer", "manufacturing_it"}]
    if primary != "manufacturing_it" and scores.get("manufacturing_it", 0) >= 2.0 and "manufacturing_it" not in secondary:
        secondary.append("manufacturing_it")
    if primary == "general_ai_data":
        secondary = []
    secondary = dedupe_preserve_order([role for role in secondary if role != primary])[:2]
    return {
        "primary": primary,
        "secondary": secondary,
        "scores": {role: round(score, 2) for role, score in scores.items() if score > 0},
        "reason": build_role_profile_reason(primary, secondary),
    }


def score_role_families(text: str) -> dict[str, float]:
    lead_text = text[:350]
    scores = {role: 0.0 for role in ROLE_HINTS}
    for role, hints in ROLE_HINTS.items():
        for hint in hints:
            if phrase_in_text(hint, text):
                scores[role] += 1.0
            if phrase_in_text(hint, lead_text):
                scores[role] += 1.5
    if phrase_in_text("software development engineer", lead_text) or phrase_in_text("full-stack software", text):
        scores["software_engineer"] += 4.0
    if phrase_in_text("ai engineer", lead_text) or phrase_in_text("machine learning engineer", lead_text):
        scores["ai_engineer"] += 4.0
    if phrase_in_text("data scientist", lead_text):
        scores["data_scientist"] += 4.0
    if phrase_in_text("smart mfg/ai", text) or phrase_in_text("smart mfg", text):
        scores["manufacturing_it"] += 2.0
        scores["ai_engineer"] -= 1.0
    if phrase_in_text("generative ai tools", text) or phrase_in_text("code generation utilities", text):
        scores["software_engineer"] += 1.0
        scores["ai_engineer"] += 0.25
    if role_has_full_stack_stack_signal(text):
        scores["software_engineer"] += 3.0
    return scores


def role_has_strong_software_signal(text: str) -> bool:
    lead_text = text[:500]
    return (
        phrase_in_text("software development engineer", lead_text)
        or phrase_in_text("software engineer", lead_text)
        or phrase_in_text("full-stack", text)
        or phrase_in_text("full stack", text)
        or role_has_full_stack_stack_signal(text)
    )


def role_has_manufacturing_system_signal(text: str) -> bool:
    return (
        phrase_in_text("mes", text)
        or phrase_in_text("manufacturing execution", text)
        or phrase_in_text("semiconductor manufacturing", text)
        or phrase_in_text("production support", text)
    )


def role_has_full_stack_product_signal(text: str) -> bool:
    return (
        phrase_in_text("full-stack", text)
        or phrase_in_text("full stack", text)
        or phrase_in_text("web technologies", text)
        or role_has_full_stack_stack_signal(text)
    )


def role_has_full_stack_stack_signal(text: str) -> bool:
    stack_terms = {
        "nodejs",
        "apache",
        "csharp",
        "dotnet",
        "angular",
        "typescript",
        "javascript",
        "api",
        "sql",
        "nosql",
        "git",
        "oop",
    }
    return len(extract_skills(text) & stack_terms) >= 4


def build_role_profile_reason(primary: str, secondary: list[str]) -> str:
    if not secondary:
        return f"Scoring uses {primary} as the primary role family."
    return (
        f"Scoring uses {primary} as the primary role family; "
        f"{', '.join(secondary)} is treated as contextual signal, not the main score target."
    )


def dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


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
        current["text"] += "\n" + line
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


def build_requirement_groups(sections: list[dict], term_details: dict[str, dict]) -> list[dict]:
    groups = []
    seen = set()
    for section in sections:
        section_type = section.get("section", "context")
        if section_type not in {"required", "responsibility", "preferred"}:
            continue
        for sentence in split_sentences(section.get("text", "")):
            sentence = normalize(sentence)
            terms = sorted(extract_skills(sentence) & set(term_details))
            if len(terms) < 2:
                continue
            operator = "any" if has_choice_marker(sentence) else "all"
            if operator != "any":
                continue
            strength = "preferred" if section_type == "preferred" else "required"
            key = (operator, strength, tuple(terms))
            if key in seen:
                continue
            seen.add(key)
            groups.append(
                {
                    "id": f"{strength}_{operator}_{len(groups) + 1}",
                    "operator": operator,
                    "strength": strength,
                    "terms": terms,
                    "text": sentence[:220],
                }
            )
    return groups


def has_choice_marker(sentence: str) -> bool:
    return any(marker in sentence for marker in {" or ", " either ", " any of ", " such as "})


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
        if any(phrase_in_text(alias, sentence) for alias in aliases) and any(marker in sentence for marker in {"required", "must", "need", "experience with"}):
            return True
    return False


def extract_skills(text: str) -> set[str]:
    found = set()
    token_set = keyword_tokens(text)
    for skill, aliases in SKILL_ALIASES.items():
        if skill == "apache" and not apache_web_context(text):
            continue
        if skill in token_set or any(phrase_in_text(alias, text) for alias in aliases):
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
        if any(phrase_in_text(alias, text) for alias in aliases):
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
    return any(phrase_in_text(alias, text) for alias in aliases) and any(marker in text for marker in {"you will", "responsible", "build", "develop", "deploy", "analyze", "communicate", "support", "troubleshoot"})


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
    if has_quantified_impact(text) or any(phrase_in_text(marker, text) for marker in {"production", "users", "deployed"}):
        return "production_or_measurable_impact"
    if category == "experience" or any(phrase_in_text(word, text) for word in {"internship", "intern", "work", "worked", "company"}):
        return "work_or_internship"
    if category == "projects":
        if any(word in text for word in {"course", "class", "homework", "assignment"}):
            return "course_or_class_project"
        return "portfolio_project"
    return "mentioned"


def has_quantified_impact(text: str) -> bool:
    return bool(re.search(r"(\d+%|\$\d+|\d+\s*(users|requests|records|rows|hours|seconds|x|ms)\b)", text))


def term_in_text(term: str, text: str) -> bool:
    if term == "apache" and not apache_web_context(text):
        return False
    aliases = SKILL_ALIASES.get(term) or RESPONSIBILITY_ALIASES.get(term) or {term}
    return term in keyword_tokens(text) or any(phrase_in_text(alias, text) for alias in aliases)


def apache_web_context(text: str) -> bool:
    normalized = normalize(text)
    if not phrase_in_text("apache", normalized):
        return False
    if any(phrase_in_text(marker, normalized) for marker in {"web service", "web server", "http", "httpd", "restful"}):
        return True
    return not any(
        phrase_in_text(marker, normalized)
        for marker in {"apache kafka", "apache iotdb", "apache spark", "apache airflow", "apache flink"}
    )


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


def requirement_units(requirements: dict, strength: str) -> list[dict]:
    terms = requirements.get(strength) or []
    groups = [
        group
        for group in requirements.get("requirement_groups", [])
        if group.get("strength") == strength and group.get("operator") == "any"
    ]
    grouped_terms = {term for group in groups for term in group.get("terms", [])}
    units = [
        {
            "operator": "any",
            "terms": list(group.get("terms", [])),
            "id": group.get("id", ""),
            "text": group.get("text", ""),
        }
        for group in groups
    ]
    units.extend({"operator": "all", "terms": [term], "id": term, "text": term} for term in terms if term not in grouped_terms)
    return [unit for unit in units if unit["terms"]]


def weighted_unit_ratio(covered: Iterable[str], units: list[dict], weights: dict[str, float]) -> float:
    if not units:
        return 0.0
    covered_set = set(covered)
    total = sum(unit_weight(unit, weights) for unit in units)
    earned = sum(unit_weight(unit, weights) for unit in units if unit_is_covered(unit, covered_set))
    return earned / total if total else 0.0


def weighted_average_unit_depth(units: list[dict], by_term: dict, weights: dict[str, float]) -> float:
    if not units:
        return 0.0
    total = sum(unit_weight(unit, weights) for unit in units)
    earned = 0.0
    for unit in units:
        terms = unit.get("terms", [])
        if unit.get("operator") == "any":
            depth = max((cv_depth(term, by_term) for term in terms), key=lambda item: DEPTH_ORDER.get(item, 0))
            earned += DEPTH_POINTS[depth] * unit_weight(unit, weights)
        else:
            earned += min(DEPTH_POINTS[cv_depth(term, by_term)] for term in terms) * unit_weight(unit, weights)
    return earned / total if total else 0.0


def unit_is_covered(unit: dict, covered_set: set[str]) -> bool:
    terms = set(unit.get("terms", []))
    if not terms:
        return False
    if unit.get("operator") == "any":
        return bool(terms & covered_set)
    return terms <= covered_set


def unit_weight(unit: dict, weights: dict[str, float]) -> float:
    values = [term_score_weight(term, weights) for term in unit.get("terms", [])]
    if not values:
        return 0.0
    if unit.get("operator") == "any":
        return max(values)
    return sum(values)


def missing_terms_for_units(requirements: dict, by_term: dict, include_preferred: bool = False) -> list[str]:
    missing = []
    strengths = ["required", "preferred"] if include_preferred else ["required"]
    for strength in strengths:
        units = requirement_units(requirements, strength)
        for unit in units:
            if unit_is_covered(unit, covered_terms(unit.get("terms", []), by_term)):
                continue
            missing.extend(term for term in unit.get("terms", []) if cv_depth(term, by_term) == "none")
    return sorted(set(missing))


def skill_gap_terms(requirements: dict, by_term: dict) -> list[str]:
    gaps = []
    for term in missing_terms_for_units(requirements, by_term, include_preferred=True):
        if supporting_depth(term, by_term) == "none":
            gaps.append(term)
    return sorted(set(gaps))


def score_cap(requirements: dict, by_term: dict) -> dict:
    if requirements.get("role_family") not in {"software_engineer", "manufacturing_it"}:
        return {"cap": None, "reason": "", "missing_groups": []}
    missing_groups = []
    for unit in requirement_units(requirements, "required"):
        terms = set(unit.get("terms", []))
        if not terms & CORE_STACK_TERMS:
            continue
        covered = credible_terms(terms, by_term)
        if unit_is_covered(unit, covered):
            continue
        missing_groups.append("/".join(unit.get("terms", [])))
    if len(missing_groups) >= 4:
        cap = 40
    elif len(missing_groups) >= 3:
        cap = 45
    elif len(missing_groups) >= 2:
        cap = 55
    else:
        cap = None
    reason = ""
    if cap is not None:
        reason = "Score capped because the current CV misses multiple core software stack requirement groups."
    return {"cap": cap, "reason": reason, "missing_groups": missing_groups}


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


def phrase_in_text(phrase: str, text: str) -> bool:
    normalized_phrase = normalize(phrase)
    normalized_text = normalize(text)
    if not normalized_phrase:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(normalized_phrase).replace(r"\ ", r"\s+") + r"(?![a-z0-9])"
    return bool(re.search(pattern, normalized_text))
