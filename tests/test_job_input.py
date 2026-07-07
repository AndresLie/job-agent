from pathlib import Path

import pytest

from career_copilot.job_input import (
    build_eightfold_api_url,
    clean_fetched_job_text,
    extract_eightfold_job_text,
    fetch_url,
    infer_company_from_url,
    is_low_quality_job_text,
    resolve_job_input,
    sanitize_filename,
    validate_fetch_url,
)


def test_resolve_job_input_from_file(tmp_path: Path):
    job = tmp_path / "job.md"
    job.write_text("# Data Scientist\nPython SQL experimentation", encoding="utf-8")
    result = resolve_job_input(job_file=job)
    assert result.source_type == "file"
    assert result.title == "Data Scientist"
    assert "Python SQL" in result.text


def test_resolve_job_input_from_text_and_cache(tmp_path: Path):
    result = resolve_job_input(
        job_text="# AI Engineer\nBuild RAG systems",
        cache_dir=tmp_path,
    )
    assert result.source_type == "text"
    assert result.title == "AI Engineer"
    assert result.cached_path is not None
    assert Path(result.cached_path).exists()


def test_resolve_job_input_from_stdin_text():
    result = resolve_job_input(use_stdin=True, stdin_text="# ML Engineer\nEvaluate models")
    assert result.source_type == "stdin"
    assert result.title == "ML Engineer"


def test_resolve_job_input_rejects_multiple_inputs(tmp_path: Path):
    job = tmp_path / "job.md"
    job.write_text("job", encoding="utf-8")
    with pytest.raises(ValueError):
        resolve_job_input(job_file=job, job_text="also job")


def test_infer_company_from_url():
    assert infer_company_from_url("https://careers.nvidia.com/jobs/123") == "Nvidia"
    assert infer_company_from_url("https://www.example-company.com/careers") == "Example Company"


def test_sanitize_filename():
    assert sanitize_filename("AI Engineer / RAG Role!") == "ai-engineer-rag-role"


def test_build_eightfold_api_url_from_pid_query():
    url = "https://careers.micron.com/careers?pid=38535468&domain=micron.com"
    assert build_eightfold_api_url(url) == "https://careers.micron.com/api/apply/v2/jobs/38535468?domain=micron.com"


def test_extract_eightfold_job_text_from_html_description():
    text = extract_eightfold_job_text(
        {
            "posting_name": "SOFTWARE DEVELOPMENT ENGINEER",
            "display_job_id": "JR84244",
            "department": "Smart MFG/AI",
            "location": "Taoyuan City, Taiwan",
            "job_description": """
            <p><b>Key Responsibilities:</b></p>
            <ul><li>Build full-stack manufacturing software.</li></ul>
            <p><b>Qualifications and Skills:</b></p>
            <ul><li>Docker, Kubernetes, OpenShift, Git, SQL.</li></ul>
            """,
        }
    )
    assert "SOFTWARE DEVELOPMENT ENGINEER" in text
    assert "Job ID: JR84244" in text
    assert "Docker, Kubernetes, OpenShift" in text
    assert not is_low_quality_job_text(text)


def test_fetch_eightfold_job_url_uses_api(monkeypatch):
    from career_copilot import job_input

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "posting_name": "SOFTWARE DEVELOPMENT ENGINEER",
                "display_job_id": "JR84244",
                "department": "Smart MFG/AI",
                "location": "Taoyuan City, Taiwan",
                "job_description": """
                <p><b>Responsibilities:</b></p>
                <ul><li>Build internal applications.</li></ul>
                <p><b>Qualifications and Skills:</b></p>
                <ul><li>Docker, Kubernetes, OpenShift, Git, SQL.</li></ul>
                """,
            }

    calls = []

    def fake_get(url, headers, timeout):
        calls.append(url)
        return Response()

    monkeypatch.setattr(job_input, "validate_fetch_url", lambda url: None)
    monkeypatch.setattr(job_input.requests, "get", fake_get)
    result = resolve_job_input(job_url="https://careers.micron.com/careers?pid=38535468&domain=micron.com")
    assert result.source_type == "url"
    assert result.title == "SOFTWARE DEVELOPMENT ENGINEER"
    assert "Docker" in result.text
    assert calls == ["https://careers.micron.com/api/apply/v2/jobs/38535468?domain=micron.com"]


def test_validate_fetch_url_blocks_localhost():
    with pytest.raises(ValueError, match="not allowed|private"):
        validate_fetch_url("http://localhost:8000/job")


def test_fetch_url_extracts_json_ld_job(monkeypatch):
    from career_copilot import job_input

    class Response:
        text = """
        <html><head><script type="application/ld+json">
        {
          "@type": "JobPosting",
          "title": "Data Scientist",
          "description": "<p>Responsibilities include SQL, Python, dashboards.</p><p>Qualifications: statistics.</p>",
          "hiringOrganization": {"name": "Example"}
        }
        </script></head><body>Shell</body></html>
        """

        def raise_for_status(self):
            return None

    monkeypatch.setattr(job_input, "validate_fetch_url", lambda url: None)
    monkeypatch.setattr(job_input.requests, "get", lambda url, headers, timeout: Response())
    page = fetch_url("https://example.com/jobs/1")
    assert page["method"] == "json_ld"
    assert page["title"] == "Data Scientist"
    assert "Python" in page["text"]


def test_fetch_url_uses_greenhouse_api(monkeypatch):
    from career_copilot import job_input

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "title": "ML Engineer",
                "location": {"name": "Remote"},
                "content": "<p>Responsibilities</p><p>Deploy Python models with Kubernetes.</p>",
            }

    calls = []
    monkeypatch.setattr(job_input, "validate_fetch_url", lambda url: None)
    monkeypatch.setattr(job_input.requests, "get", lambda url, headers, timeout: calls.append(url) or Response())
    page = fetch_url("https://boards.greenhouse.io/acme/jobs/12345")
    assert page["method"] == "greenhouse_api"
    assert "boards-api.greenhouse.io" in calls[0]
    assert "Kubernetes" in page["text"]


def test_fetch_url_uses_lever_api(monkeypatch):
    from career_copilot import job_input

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "text": "AI Engineer",
                "descriptionPlain": "Build RAG systems.",
                "lists": [{"text": "Requirements", "content": [{"text": "Python and evaluation."}]}],
            }

    calls = []
    monkeypatch.setattr(job_input, "validate_fetch_url", lambda url: None)
    monkeypatch.setattr(job_input.requests, "get", lambda url, headers, timeout: calls.append(url) or Response())
    page = fetch_url("https://jobs.lever.co/acme/abc123")
    assert page["method"] == "lever_api"
    assert "api.lever.co" in calls[0]
    assert "evaluation" in page["text"]


def test_clean_fetched_job_text_removes_embedded_config():
    raw = """
    Software Engineer
    {"navbarData": {"customHtmlNavbarData": {"css": ".container{display:flex}"}}}
    Responsibilities
    Build Python services with Docker and SQL.
    Requirements
    Experience with APIs and deployment.
    """
    text = clean_fetched_job_text(raw)
    assert "navbarData" not in text
    assert "container" not in text
    assert "Docker" in text
    assert not is_low_quality_job_text(text)


def test_low_quality_job_text_detects_dynamic_shell():
    text = clean_fetched_job_text(
        "SOFTWARE DEVELOPMENT ENGINEER\n"
        '{"themeOptions": {"navbarData": {"css": ".container{display:flex}"}}}'
    )
    assert is_low_quality_job_text(text)


def test_resolve_job_url_rejects_dynamic_shell(monkeypatch):
    from career_copilot import job_input

    monkeypatch.setattr(
        job_input,
        "fetch_url",
        lambda url: {
            "title": "Dynamic Careers Page",
            "text": 'SOFTWARE DEVELOPMENT ENGINEER\n{"navbarData": {"css": ".container{display:flex}"}}',
        },
    )
    with pytest.raises(ValueError, match="paste the JD text"):
        resolve_job_input(job_url="https://careers.example.com/job")
