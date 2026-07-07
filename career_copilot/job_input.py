from __future__ import annotations

import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests

from .documents import clean_text


@dataclass(frozen=True)
class JobInput:
    source_type: str
    text: str
    title: str
    company: str
    url: str | None = None
    cached_path: str | None = None


def resolve_job_input(
    *,
    job_file: Path | None = None,
    job_text: str | None = None,
    job_url: str | None = None,
    use_stdin: bool = False,
    company: str | None = None,
    cache_dir: Path | None = None,
    cache: bool = True,
    stdin_text: str | None = None,
) -> JobInput:
    selected = [
        value is not None and str(value).strip() != ""
        for value in (job_file, job_text, job_url)
    ] + [use_stdin]
    if sum(1 for item in selected if item) != 1:
        raise ValueError("Provide exactly one job input: --job-file/--job, --job-text, --job-url, or --stdin.")

    if job_file is not None:
        text = clean_text(job_file.read_text(encoding="utf-8", errors="ignore"))
        return JobInput(
            source_type="file",
            text=text,
            title=infer_title(text, job_file.stem),
            company=company or "",
            cached_path=str(job_file),
        )

    if job_text:
        text = clean_text(job_text)
        cached_path = cache_text(text, cache_dir, "pasted-job", "text") if cache and cache_dir else None
        return JobInput(
            source_type="text",
            text=text,
            title=infer_title(text, "Pasted Job Description"),
            company=company or "",
            cached_path=cached_path,
        )

    if use_stdin:
        raw = stdin_text if stdin_text is not None else sys.stdin.read()
        text = clean_text(raw)
        cached_path = cache_text(text, cache_dir, "stdin-job", "stdin") if cache and cache_dir else None
        return JobInput(
            source_type="stdin",
            text=text,
            title=infer_title(text, "Stdin Job Description"),
            company=company or "",
            cached_path=cached_path,
        )

    assert job_url is not None
    page = fetch_url(job_url)
    text = clean_fetched_job_text(page["text"])
    if is_low_quality_job_text(text):
        raise ValueError(
            "Could not extract a clean job description from this URL. "
            "The page may be dynamic or protected; paste the JD text instead."
        )
    inferred_company = company or infer_company_from_url(job_url)
    cached_path = cache_text(text, cache_dir, inferred_company or "job-url", "url", source_url=job_url) if cache and cache_dir else None
    return JobInput(
        source_type="url",
        text=text,
        title=infer_title(text, page.get("title") or "Fetched Job Description"),
        company=inferred_company,
        url=job_url,
        cached_path=cached_path,
    )


def fetch_url(url: str) -> dict[str, str]:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError("Install beautifulsoup4 to use --job-url.") from exc

    try:
        response = requests.get(
            url,
            headers={"User-Agent": "ai-job-copilot/0.1"},
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not fetch job URL: {exc}") from exc
    soup = BeautifulSoup(response.text, "html.parser")
    for element in soup(["script", "style", "noscript"]):
        element.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    text = soup.get_text("\n", strip=True)
    return {"title": title, "text": text}


def clean_fetched_job_text(text: str) -> str:
    lines = []
    for line in clean_text(text).splitlines():
        stripped = line.strip()
        lowered = stripped.casefold()
        if not stripped:
            continue
        if looks_like_embedded_config(stripped, lowered):
            continue
        lines.append(stripped)
    return clean_text("\n".join(lines))


def looks_like_embedded_config(stripped: str, lowered: str) -> bool:
    if stripped.startswith(("{", "[")) and len(stripped) > 200:
        return True
    if len(stripped) > 1000 and any(marker in lowered for marker in {"navbar", "themeoptions", "css", "scripts"}):
        return True
    return any(
        marker in lowered
        for marker in {
            "themeoptions",
            "navbardata",
            "customhtmlnavbardata",
            "scriptconfig",
            "notificationkeyratelimit",
            "platformperformance",
        }
    )


def is_low_quality_job_text(text: str) -> bool:
    tokens = re.findall(r"[a-zA-Z]{3,}", text)
    lowered = text.casefold()
    job_markers = {
        "responsibilities",
        "requirements",
        "qualifications",
        "minimum qualifications",
        "preferred qualifications",
        "job description",
        "what you will do",
        "about the role",
        "skills",
    }
    has_marker = any(marker in lowered for marker in job_markers)
    if len(tokens) < 10:
        return True
    if len(tokens) < 80 and not has_marker:
        return True
    return not has_marker


def infer_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip("# ").strip()
        if stripped and len(stripped) <= 90:
            return stripped
    return fallback.replace("_", " ").replace("-", " ").title()


def infer_company_from_url(url: str) -> str:
    host = urlparse(url).netloc.casefold()
    host = re.sub(r"^www\.", "", host)
    parts = [part for part in host.split(".") if part not in {"jobs", "careers", "boards", "apply"}]
    if not parts:
        return ""
    return parts[0].replace("-", " ").title()


def cache_text(
    text: str,
    cache_dir: Path | None,
    label: str,
    source_type: str,
    source_url: str | None = None,
) -> str | None:
    if cache_dir is None:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1((source_url or text).encode("utf-8")).hexdigest()[:10]
    path = cache_dir / f"{sanitize_filename(label)}-{source_type}-{digest}.md"
    header = [f"# Cached Job Source: {label}", "", f"- Source type: {source_type}"]
    if source_url:
        header.append(f"- URL: {source_url}")
    path.write_text("\n".join(header) + "\n\n" + text + "\n", encoding="utf-8")
    return str(path)


def sanitize_filename(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug[:60] or "job"
