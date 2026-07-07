from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import socket
import sys
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from urllib.parse import parse_qs, urlparse

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
    extraction_method: str = "direct"


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
        extraction_method=page.get("method", "html"),
    )


def fetch_url(url: str) -> dict[str, str]:
    validate_fetch_url(url)
    if is_linkedin_job_url(url):
        raise ValueError("LinkedIn job pages usually block automated extraction. Paste the JD text instead.")
    eightfold_page = fetch_eightfold_job(url)
    if eightfold_page:
        return eightfold_page
    workday_page = fetch_workday_job(url)
    if workday_page:
        return workday_page
    ashby_page = fetch_ashby_job(url)
    if ashby_page:
        return ashby_page
    smartrecruiters_page = fetch_smartrecruiters_job(url)
    if smartrecruiters_page:
        return smartrecruiters_page
    greenhouse_page = fetch_greenhouse_job(url)
    if greenhouse_page:
        return greenhouse_page
    lever_page = fetch_lever_job(url)
    if lever_page:
        return lever_page

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
    json_ld_page = extract_json_ld_job(soup)
    if json_ld_page:
        return json_ld_page
    for element in soup(["script", "style", "noscript"]):
        element.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    text = soup.get_text("\n", strip=True)
    return {"title": title, "text": text, "method": "html"}


def validate_fetch_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Job URL must be an http or https URL.")
    host = parsed.hostname or ""
    if host.casefold() in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        raise ValueError("Job URL host is not allowed.")
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)}
    except socket.gaierror as exc:
        raise RuntimeError(f"Could not resolve job URL host: {exc}") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
            raise ValueError("Job URL host resolves to a private or local network address.")


def fetch_eightfold_job(url: str) -> dict[str, str] | None:
    api_url = build_eightfold_api_url(url)
    if not api_url:
        return None
    validate_fetch_url(api_url)

    try:
        response = requests.get(
            api_url,
            headers={
                "User-Agent": "ai-job-copilot/0.1",
                "Accept": "application/json",
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return None

    if not isinstance(data, dict):
        return None
    text = extract_eightfold_job_text(data)
    if not text:
        return None
    title = str(data.get("posting_name") or data.get("name") or "Fetched Job Description")
    return {"title": title, "text": text, "method": "eightfold_api"}


def build_eightfold_api_url(url: str) -> str | None:
    parsed = urlparse(url)
    pid = extract_eightfold_pid(parsed)
    if not pid or not parsed.scheme or not parsed.netloc:
        return None
    domain = extract_eightfold_domain(parsed)
    api_url = f"{parsed.scheme}://{parsed.netloc}/api/apply/v2/jobs/{pid}"
    if domain:
        api_url = f"{api_url}?domain={domain}"
    return api_url


def extract_eightfold_pid(parsed_url) -> str | None:
    query = parse_qs(parsed_url.query)
    if query.get("pid") and query["pid"][0].isdigit():
        return query["pid"][0]
    match = re.search(r"/careers/job/(\d+)", parsed_url.path)
    if match:
        return match.group(1)
    return None


def extract_eightfold_domain(parsed_url) -> str:
    query = parse_qs(parsed_url.query)
    if query.get("domain"):
        return query["domain"][0]
    host = parsed_url.netloc.casefold()
    host = re.sub(r"^www\.", "", host)
    labels = [label for label in host.split(".") if label not in {"careers", "jobs", "apply"}]
    if len(labels) >= 2 and labels[-2:] != ["eightfold", "ai"]:
        return ".".join(labels[-2:])
    return host


def extract_eightfold_job_text(data: dict) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError("Install beautifulsoup4 to use --job-url.") from exc

    parts = [
        str(data.get("posting_name") or data.get("name") or ""),
        f"Job ID: {data.get('display_job_id') or data.get('ats_job_id')}" if data.get("display_job_id") or data.get("ats_job_id") else "",
        f"Department: {data.get('department')}" if data.get("department") else "",
        f"Location: {data.get('location')}" if data.get("location") else "",
    ]
    description = data.get("job_description") or data.get("description") or ""
    if description:
        soup = BeautifulSoup(str(description), "html.parser")
        parts.append(soup.get_text("\n", strip=True))
    return clean_text("\n".join(part for part in parts if part))


def fetch_greenhouse_job(url: str) -> dict[str, str] | None:
    parsed = urlparse(url)
    if not host_matches(parsed.netloc, "greenhouse.io"):
        return None
    match = re.search(r"/([^/]+)/jobs/(\d+)", parsed.path)
    if not match:
        return None
    board, job_id = match.groups()
    api_url = f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs/{job_id}?content=true"
    validate_fetch_url(api_url)
    try:
        response = requests.get(api_url, headers={"User-Agent": "ai-job-copilot/0.1", "Accept": "application/json"}, timeout=30)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return None
    content = data.get("content") or ""
    title = str(data.get("title") or "Fetched Job Description")
    location = ""
    if isinstance(data.get("location"), dict):
        location = data["location"].get("name") or ""
    text = html_to_text("\n".join(part for part in [title, location, content] if part))
    return {"title": title, "text": text, "method": "greenhouse_api"} if text else None


def fetch_lever_job(url: str) -> dict[str, str] | None:
    parsed = urlparse(url)
    if not host_matches(parsed.netloc, "lever.co"):
        return None
    match = re.search(r"/([^/]+)/([^/?#]+)", parsed.path)
    if not match:
        return None
    company, posting_id = match.groups()
    api_url = f"https://api.lever.co/v0/postings/{company}/{posting_id}"
    validate_fetch_url(api_url)
    try:
        response = requests.get(api_url, headers={"User-Agent": "ai-job-copilot/0.1", "Accept": "application/json"}, timeout=30)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return None
    title = str(data.get("text") or "Fetched Job Description")
    parts = [title, data.get("descriptionPlain") or html_to_text(str(data.get("description") or ""))]
    for section in data.get("lists") or []:
        if not isinstance(section, dict):
            continue
        parts.append(str(section.get("text") or ""))
        for item in section.get("content") or []:
            parts.append(str(item.get("text") if isinstance(item, dict) else item))
    text = clean_text("\n".join(part for part in parts if part))
    return {"title": title, "text": text, "method": "lever_api"} if text else None


def fetch_workday_job(url: str) -> dict[str, str] | None:
    parsed = urlparse(url)
    if "myworkdayjobs.com" not in parsed.netloc:
        return None
    api_url = build_workday_api_url(url)
    if not api_url:
        return None
    validate_fetch_url(api_url)
    try:
        response = requests.get(api_url, headers={"User-Agent": "ai-job-copilot/0.1", "Accept": "application/json"}, timeout=30)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return None
    info = data.get("jobPostingInfo") if isinstance(data.get("jobPostingInfo"), dict) else data
    title = str(info.get("title") or info.get("jobPostingTitle") or "Fetched Job Description")
    parts = [
        title,
        str(info.get("location") or ""),
        str(info.get("jobDescription") or ""),
        str(info.get("qualifications") or ""),
        str(info.get("responsibilities") or ""),
    ]
    text = html_to_text("\n".join(part for part in parts if part))
    return {"title": title, "text": text, "method": "workday_api"} if text else None


def build_workday_api_url(url: str) -> str | None:
    parsed = urlparse(url)
    host = parsed.netloc
    tenant = host.split(".", 1)[0].split("-")[0]
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) < 3:
        return None
    if re.fullmatch(r"[a-z]{2}-[A-Z]{2}", parts[0]) and len(parts) >= 4:
        parts = parts[1:]
    site = parts[0]
    if "job" not in parts:
        return None
    job_index = parts.index("job")
    job_path = "/".join(parts[job_index:])
    return f"{parsed.scheme}://{host}/wday/cxs/{tenant}/{site}/{job_path}"


def fetch_ashby_job(url: str) -> dict[str, str] | None:
    parsed = urlparse(url)
    if not host_matches(parsed.netloc, "ashbyhq.com"):
        return None
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) < 2:
        return None
    organization, posting_id = parts[0], parts[-1]
    api_url = f"{parsed.scheme}://{parsed.netloc}/api/non-user-graphql?op=ApiJobPosting"
    validate_fetch_url(api_url)
    payload = {
        "operationName": "ApiJobPosting",
        "variables": {
            "organizationHostedJobsPageName": organization,
            "jobPostingId": posting_id,
        },
        "query": (
            "query ApiJobPosting($organizationHostedJobsPageName: String!, $jobPostingId: String!) { "
            "jobPosting(organizationHostedJobsPageName: $organizationHostedJobsPageName, jobPostingId: $jobPostingId) { "
            "title descriptionPlain description locationName departmentName } }"
        ),
    }
    try:
        response = requests.post(api_url, headers={"User-Agent": "ai-job-copilot/0.1", "Accept": "application/json"}, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return None
    posting = ((data.get("data") or {}).get("jobPosting") if isinstance(data, dict) else None) or {}
    if not isinstance(posting, dict):
        return None
    title = str(posting.get("title") or "Fetched Job Description")
    text = clean_text(
        "\n".join(
            part
            for part in [
                title,
                str(posting.get("locationName") or ""),
                str(posting.get("departmentName") or ""),
                posting.get("descriptionPlain") or html_to_text(str(posting.get("description") or "")),
            ]
            if part
        )
    )
    return {"title": title, "text": text, "method": "ashby_api"} if text else None


def fetch_smartrecruiters_job(url: str) -> dict[str, str] | None:
    parsed = urlparse(url)
    if not host_matches(parsed.netloc, "smartrecruiters.com"):
        return None
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) < 2:
        return None
    company, posting_id = parts[0], parts[-1]
    api_url = f"https://api.smartrecruiters.com/v1/companies/{company}/postings/{posting_id}"
    validate_fetch_url(api_url)
    try:
        response = requests.get(api_url, headers={"User-Agent": "ai-job-copilot/0.1", "Accept": "application/json"}, timeout=30)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return None
    title = str(data.get("name") or "Fetched Job Description")
    sections = []
    job_ad = data.get("jobAd") if isinstance(data.get("jobAd"), dict) else {}
    for section in job_ad.get("sections") or {}:
        value = job_ad.get("sections", {}).get(section)
        if value:
            sections.append(str(value))
    text = html_to_text("\n".join([title, str(data.get("location", {}).get("city") if isinstance(data.get("location"), dict) else ""), *sections]))
    return {"title": title, "text": text, "method": "smartrecruiters_api"} if text else None


def extract_json_ld_job(soup) -> dict[str, str] | None:
    for script in soup.find_all("script", attrs={"type": re.compile("ld\\+json", re.I)}):
        raw = script.string or script.get_text(" ", strip=True)
        if not raw:
            continue
        for item in iter_json_objects(raw):
            job = find_job_posting(item)
            if not job:
                continue
            title = str(job.get("title") or job.get("name") or "Fetched Job Description")
            org = job.get("hiringOrganization") if isinstance(job.get("hiringOrganization"), dict) else {}
            location = format_job_location(job.get("jobLocation"))
            parts = [
                title,
                str(org.get("name") or ""),
                location,
                html_to_text(str(job.get("description") or "")),
                html_to_text(str(job.get("responsibilities") or "")),
                html_to_text(str(job.get("qualifications") or "")),
            ]
            text = clean_text("\n".join(part for part in parts if part))
            if text:
                return {"title": title, "text": text, "method": "json_ld"}
    return None


def iter_json_objects(raw: str):
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    return []


def find_job_posting(item):
    if not isinstance(item, dict):
        return None
    item_type = item.get("@type")
    types = item_type if isinstance(item_type, list) else [item_type]
    if any(str(value).casefold() == "jobposting" for value in types):
        return item
    graph = item.get("@graph")
    if isinstance(graph, list):
        for node in graph:
            found = find_job_posting(node)
            if found:
                return found
    return None


def format_job_location(value) -> str:
    locations = value if isinstance(value, list) else [value]
    parts = []
    for location in locations:
        if not isinstance(location, dict):
            continue
        address = location.get("address")
        if isinstance(address, dict):
            parts.append(
                ", ".join(
                    str(address.get(key))
                    for key in ("addressLocality", "addressRegion", "addressCountry")
                    if address.get(key)
                )
            )
    return "; ".join(part for part in parts if part)


def html_to_text(value: str) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError("Install beautifulsoup4 to use --job-url.") from exc
    return clean_text(unescape(BeautifulSoup(value, "html.parser").get_text("\n", strip=True)))


def host_matches(host: str, suffix: str) -> bool:
    normalized = host.casefold().split(":", 1)[0]
    return normalized == suffix or normalized.endswith(f".{suffix}")


def is_linkedin_job_url(url: str) -> bool:
    parsed = urlparse(url)
    return host_matches(parsed.netloc, "linkedin.com") and "/jobs/" in parsed.path


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
