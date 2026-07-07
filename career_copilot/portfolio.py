from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .documents import chunk_document, discover_documents, load_document
from .rubric import known_skill_terms
from .vector_store import keyword_tokens


ROLE_TARGETS = {
    "data_scientist": {"python", "sql", "statistics", "pandas", "dashboard", "forecasting", "ab_test"},
    "ai_engineer": {"python", "rag", "llm", "agent", "embedding", "retrieval", "evaluation", "vector_search"},
    "ml_engineer": {"python", "machine_learning", "deployment", "mlops", "kubernetes", "docker", "api"},
}


def build_portfolio_report(source: Path) -> dict[str, Any]:
    docs = discover_documents(source)
    root = source if source.is_dir() else source.parent
    items = []
    all_terms: Counter[str] = Counter()
    for doc in docs:
        rel = doc.resolve().relative_to(root.resolve()).as_posix()
        if not rel.split("/", 1)[0].casefold() in {"projects", "project", "experience", "experiences", "work"}:
            continue
        text = load_document(doc)
        chunks = chunk_document(doc, text, root=root)
        terms = sorted(extract_portfolio_terms(text))
        all_terms.update(terms)
        items.append(
            {
                "source_path": rel,
                "category": chunks[0].category if chunks else "general",
                "terms": terms,
                "summary": " ".join(text.split())[:260],
            }
        )
    role_reports = {}
    for role, targets in ROLE_TARGETS.items():
        covered = sorted(targets & set(all_terms))
        missing = sorted(targets - set(all_terms))
        role_reports[role] = {
            "coverage": round(len(covered) / len(targets), 3),
            "covered_terms": covered,
            "missing_terms": missing,
            "strongest_sources": strongest_sources(items, targets),
            "recommended_next_project": recommend_project(role, missing),
        }
    return {
        "documents": len(items),
        "top_terms": all_terms.most_common(20),
        "roles": role_reports,
        "items": items,
    }


def export_portfolio_report(source: Path, output_path: Path) -> Path:
    report = build_portfolio_report(source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".json":
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        output_path.write_text(render_portfolio_markdown(report), encoding="utf-8")
    return output_path


def extract_portfolio_terms(text: str) -> set[str]:
    tokens = keyword_tokens(text)
    terms = set()
    for term in known_skill_terms():
        normalized = term.replace("_", " ")
        if term in tokens or normalized in text.casefold():
            terms.add(term)
    return terms


def strongest_sources(items: list[dict[str, Any]], targets: set[str]) -> list[dict[str, Any]]:
    ranked = []
    for item in items:
        overlap = sorted(set(item["terms"]) & targets)
        if overlap:
            ranked.append({"source_path": item["source_path"], "matched_terms": overlap, "count": len(overlap)})
    ranked.sort(key=lambda item: (-item["count"], item["source_path"]))
    return ranked[:5]


def recommend_project(role: str, missing: list[str]) -> str:
    if not missing:
        return f"Portfolio coverage for {role} is strong; improve quantified impact and deployment evidence."
    focus = ", ".join(missing[:4])
    return f"Build or document one {role.replace('_', ' ')} project that demonstrates: {focus}."


def render_portfolio_markdown(report: dict[str, Any]) -> str:
    lines = ["# Portfolio Report", "", f"- Documents analyzed: {report['documents']}", ""]
    lines.append("## Role Coverage")
    for role, data in report["roles"].items():
        lines.extend(
            [
                "",
                f"### {role.replace('_', ' ').title()}",
                "",
                f"- Coverage: {data['coverage']}",
                f"- Covered: {', '.join(data['covered_terms']) or 'none'}",
                f"- Missing: {', '.join(data['missing_terms']) or 'none'}",
                f"- Next project: {data['recommended_next_project']}",
                "",
                "Strongest sources:",
            ]
        )
        for source in data["strongest_sources"]:
            lines.append(f"- `{source['source_path']}`: {', '.join(source['matched_terms'])}")
    lines.extend(["", "## Top Terms"])
    for term, count in report["top_terms"]:
        lines.append(f"- {term}: {count}")
    return "\n".join(lines).rstrip() + "\n"
