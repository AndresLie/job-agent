from __future__ import annotations

import re

from .rubric import RESPONSIBILITY_ALIASES, SKILL_ALIASES


MIGRATION_PATTERNS = [
    r"\bmigrat\w*\b",
    r"\bmoderni[sz]\w*\b",
    r"\bmoving\s+(?:from|to|toward)\b",
    r"\bmove\s+(?:from|to|toward)\b",
    r"\blegacy\b",
    r"\breplatform\w*\b",
    r"\btransition\w*\s+(?:from|to)\b",
]


def build_company_context_review(
    *,
    requirements: dict,
    web_research: list[dict] | None,
    company: str = "",
    role: str = "",
    job_text: str = "",
) -> dict:
    sources = normalize_sources(web_research or [])
    if not sources:
        return {
            "status": "no_external_sources",
            "company": company,
            "role": role,
            "summary": build_summary(
                status="no_external_sources",
                corroborated=[],
                weak_external=[],
                migration_signals=[],
            ),
            "score_policy": "advisory_only_no_score_change",
            "sources_considered": 0,
            "signals": [],
            "weight_recommendations": [],
            "migration_signals": [],
            "uncertainties": build_uncertainties([], [], []),
        }
    terms = requirement_terms(requirements)
    source_matches = {term: source_term_matches(term, sources) for term in terms}
    jd_counts = {term: count_term(term, job_text) for term in terms}
    recommendations = [
        build_weight_recommendation(term, current_strength(term, requirements), jd_counts[term], source_matches[term])
        for term in terms
    ]
    signals = [
        build_signal(term, current_strength(term, requirements), jd_counts[term], source_matches[term])
        for term in terms
    ]
    migration_signals = detect_migration_signals(sources)
    uncertainties = build_uncertainties(sources, recommendations, migration_signals)
    corroborated = [item["term"] for item in recommendations if item["external_source_count"] > 0]
    weak_external = [
        item["term"]
        for item in recommendations
        if item["recommendation"] in {"uncertain", "consider_downweight"}
    ]
    status = "used" if sources else "no_external_sources"
    summary = build_summary(
        status=status,
        corroborated=corroborated,
        weak_external=weak_external,
        migration_signals=migration_signals,
    )
    return {
        "status": status,
        "company": company,
        "role": role,
        "summary": summary,
        "score_policy": "advisory_only_no_score_change",
        "sources_considered": len(sources),
        "signals": signals,
        "weight_recommendations": recommendations,
        "migration_signals": migration_signals,
        "uncertainties": uncertainties,
    }


def normalize_sources(web_research: list[dict]) -> list[dict]:
    sources = []
    for index, source in enumerate(web_research):
        if not isinstance(source, dict):
            continue
        highlights = source.get("highlights") or []
        if isinstance(highlights, str):
            highlights = [highlights]
        parts = [
            str(source.get("title") or ""),
            str(source.get("summary") or ""),
            " ".join(str(item) for item in highlights),
        ]
        text = normalize_text(" ".join(parts))
        if not text.strip():
            continue
        sources.append(
            {
                "index": index,
                "title": str(source.get("title") or "Untitled"),
                "url": str(source.get("url") or ""),
                "source_type": str(source.get("source_type") or "web"),
                "text": text,
                "raw_text": " ".join(parts),
            }
        )
    return sources


def requirement_terms(requirements: dict) -> list[str]:
    ordered = []
    seen = set()
    for key in ("required", "preferred", "context", "responsibilities"):
        for term in requirements.get(key, []) or []:
            value = str(term).strip()
            if value and value not in seen:
                ordered.append(value)
                seen.add(value)
    return ordered


def current_strength(term: str, requirements: dict) -> str:
    for key in ("required", "preferred", "context", "responsibilities", "ignored"):
        if term in set(requirements.get(key, []) or []):
            return "responsibility" if key == "responsibilities" else key
    return "unknown"


def source_term_matches(term: str, sources: list[dict]) -> list[dict]:
    matches = []
    variants = term_variants(term)
    for source in sources:
        count = sum(count_phrase(variant, source["text"]) for variant in variants)
        if count:
            matches.append(
                {
                    "title": source["title"],
                    "url": source["url"],
                    "source_type": source["source_type"],
                    "count": count,
                }
            )
    return matches


def build_weight_recommendation(term: str, strength: str, jd_count: int, matches: list[dict]) -> dict:
    external_count = len(matches)
    source_urls = [item["url"] for item in matches if item.get("url")][:3]
    if external_count >= 2 and strength in {"required", "responsibility"}:
        recommendation = "keep_high"
        suggested_strength = strength
        confidence = "high"
        reason = f"{term} appears in the JD and is corroborated by {external_count} external source(s)."
    elif external_count >= 1 and strength in {"required", "responsibility"}:
        recommendation = "keep_high"
        suggested_strength = strength
        confidence = "medium"
        reason = f"{term} appears in the JD and has at least one external corroborating source."
    elif external_count >= 2 and strength in {"preferred", "context", "ignored"}:
        recommendation = "consider_promote"
        suggested_strength = "preferred"
        confidence = "medium"
        reason = f"{term} is weakly weighted in the JD parse but appears across external company context."
    elif external_count == 0 and strength in {"required", "responsibility"} and jd_count <= 1:
        recommendation = "uncertain"
        suggested_strength = strength
        confidence = "low"
        reason = f"{term} is treated as important from the JD, but external sources did not corroborate it."
    elif external_count == 0 and strength in {"preferred", "context"}:
        recommendation = "consider_downweight"
        suggested_strength = "context"
        confidence = "low"
        reason = f"{term} appears in the JD parse, but external sources did not reinforce it."
    else:
        recommendation = "keep"
        suggested_strength = strength
        confidence = "medium" if external_count else "low"
        reason = f"{term} has no strong external reason to change its parsed weight."
    return {
        "term": term,
        "current_strength": strength,
        "recommendation": recommendation,
        "suggested_strength": suggested_strength,
        "confidence": confidence,
        "reason": reason,
        "jd_mentions": jd_count,
        "external_source_count": external_count,
        "source_urls": source_urls,
    }


def build_signal(term: str, strength: str, jd_count: int, matches: list[dict]) -> dict:
    if not matches:
        signal = "jd_only" if jd_count else "not_supported_by_external_sources"
        confidence = "low"
        reason = f"{term} was not found in external company context."
    elif len(matches) == 1:
        signal = "single_external_source"
        confidence = "medium"
        reason = f"{term} was found in one external source."
    else:
        signal = "repeated_in_external_sources"
        confidence = "high"
        reason = f"{term} was found in {len(matches)} external source(s)."
    return {
        "term": term,
        "current_strength": strength,
        "signal": signal,
        "confidence": confidence,
        "reason": reason,
        "source_urls": [item["url"] for item in matches if item.get("url")][:3],
    }


def build_uncertainties(sources: list[dict], recommendations: list[dict], migration_signals: list[dict]) -> list[str]:
    uncertainties = []
    if not sources:
        uncertainties.append("No external company sources were available; weighting review is based on JD parsing only.")
        return uncertainties
    if not migration_signals:
        uncertainties.append("No explicit migration or stack-change signal was found in the external sources.")
    weak = [
        item["term"]
        for item in recommendations
        if item["current_strength"] in {"required", "responsibility"} and item["external_source_count"] == 0
    ]
    if weak:
        uncertainties.append(
            "These important JD terms lacked external corroboration and should be checked manually: "
            + ", ".join(weak[:8])
            + "."
        )
    return uncertainties


def build_summary(
    *,
    status: str,
    corroborated: list[str],
    weak_external: list[str],
    migration_signals: list[dict],
) -> str:
    if status != "used":
        return "No external company sources were used. Requirement weights remain the JD-only parse and the score is unchanged."
    parts = []
    if corroborated:
        parts.append("External sources reinforce: " + ", ".join(corroborated[:6]) + ".")
    if weak_external:
        parts.append("External corroboration is weak or absent for: " + ", ".join(weak_external[:6]) + ".")
    if migration_signals:
        parts.append("Possible stack-change or migration signals were detected and should be verified before reweighting.")
    parts.append("These findings are advisory only and do not change the CV-to-JD score.")
    return " ".join(parts)


def detect_migration_signals(sources: list[dict]) -> list[dict]:
    signals = []
    for source in sources:
        text = source["raw_text"]
        for sentence in split_sentences(text):
            normalized = normalize_text(sentence)
            if any(re.search(pattern, normalized) for pattern in MIGRATION_PATTERNS):
                signals.append(
                    {
                        "source_title": source["title"],
                        "source_url": source["url"],
                        "excerpt": trim(sentence, 220),
                    }
                )
                break
    return signals[:5]


def term_variants(term: str) -> set[str]:
    normalized = str(term).strip().casefold()
    variants = {normalized, normalized.replace("_", " "), normalized.replace("-", " ")}
    variants |= SKILL_ALIASES.get(normalized, set())
    variants |= RESPONSIBILITY_ALIASES.get(normalized, set())
    if normalized == "csharp":
        variants |= {"c#", "c sharp", ".net", "dotnet"}
    if normalized == "production_support":
        variants |= {"support", "supporting production", "production issue", "operational support"}
    if normalized == "troubleshooting":
        variants |= {"root cause", "debug", "debugging", "problem resolution"}
    return {normalize_text(item) for item in variants if item}


def count_term(term: str, text: str) -> int:
    normalized = normalize_text(text)
    return sum(count_phrase(variant, normalized) for variant in term_variants(term))


def count_phrase(phrase: str, text: str) -> int:
    if not phrase:
        return 0
    escaped = re.escape(phrase)
    if re.match(r"^[a-z0-9 ]+$", phrase):
        return len(re.findall(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", text))
    return text.count(phrase)


def normalize_text(value: str) -> str:
    text = str(value).casefold()
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    return [item.strip() for item in re.split(r"(?<=[.!?])\s+", str(text)) if item.strip()]


def trim(value: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."
