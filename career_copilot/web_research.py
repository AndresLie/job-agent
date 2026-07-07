from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass(frozen=True)
class WebSource:
    title: str
    url: str
    source_type: str
    highlights: list[str]
    summary: str


def research_company(
    *,
    company: str,
    role: str = "AI Engineer",
    num_results: int = 5,
    cache_dir: Path | None = None,
    cache: bool = True,
) -> list[dict]:
    api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        raise RuntimeError("EXA_API_KEY is required for --research-company.")

    try:
        from exa_py import Exa
    except ImportError as exc:
        raise RuntimeError("Install exa-py to use company research.") from exc

    query = (
        f"{company} {role} engineering culture AI data machine learning "
        "company product recent interview preparation"
    )
    exa = Exa(api_key=api_key)
    response = exa.search(
        query,
        type="auto",
        num_results=num_results,
        contents={"highlights": True},
    )
    sources = normalize_exa_results(response)
    if cache:
        cache_research(company, role, sources, cache_dir)
    return [asdict(source) for source in sources]


def normalize_exa_results(response) -> list[WebSource]:
    raw_results = getattr(response, "results", response)
    sources = []
    for item in raw_results or []:
        title = get_value(item, "title") or "Untitled"
        url = get_value(item, "url") or ""
        highlights = get_value(item, "highlights") or []
        if isinstance(highlights, str):
            highlights = [highlights]
        summary = get_value(item, "summary") or " ".join(str(value) for value in highlights[:2])
        sources.append(
            WebSource(
                title=str(title),
                url=str(url),
                source_type="exa",
                highlights=[str(value) for value in highlights[:4]],
                summary=str(summary)[:600],
            )
        )
    return sources


def get_value(item, name: str):
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def cache_research(
    company: str,
    role: str,
    sources: list[WebSource],
    cache_dir: Path | None,
) -> str | None:
    if cache_dir is None:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "company": company,
        "role": role,
        "sources": [asdict(source) for source in sources],
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    path = cache_dir / f"{sanitize_filename(company)}-{sanitize_filename(role)}-{digest}.md"
    lines = [f"# Company Research: {company}", "", f"Role: {role}", ""]
    for index, source in enumerate(sources, start=1):
        lines.extend(
            [
                f"## Source {index}: {source.title}",
                f"URL: {source.url}",
                "",
                source.summary,
                "",
            ]
        )
        for highlight in source.highlights:
            lines.append(f"- {highlight}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def sanitize_filename(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug[:60] or "research"
