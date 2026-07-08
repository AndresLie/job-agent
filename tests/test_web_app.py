from pathlib import Path

from fastapi.testclient import TestClient

from career_copilot.web_app import create_app


def build_example_root(tmp_path: Path) -> Path:
    examples = tmp_path / "examples"
    for folder, filename, text in [
        ("resume", "cv.md", "Deployed Python SQL RAG retrieval evaluation work and improved coverage by 30%."),
        ("projects", "rag.md", "Built a RAG retrieval evaluation portfolio project with citations."),
        ("experience", "internship.md", "Delivered machine learning evaluation pipelines during an internship."),
    ]:
        path = examples / folder / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return tmp_path


def test_web_health(tmp_path, monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    client = TestClient(create_app(project_root=build_example_root(tmp_path)))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_web_index_renders_form(tmp_path, monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    client = TestClient(create_app(project_root=build_example_root(tmp_path)))
    response = client.get("/")
    assert response.status_code == 200
    assert "Run Review" in response.text
    assert "Job description" in response.text
    assert "data/raw" in response.text or "data\\raw" in response.text
    assert "CV vs JD Review" in response.text or "No review yet" in response.text
    assert "data-review-form" in response.text
    assert "data-submit-status" in response.text
    assert "/static/app.js" in response.text


def test_web_review_with_pasted_jd_returns_result(tmp_path, monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    root = build_example_root(tmp_path)
    client = TestClient(create_app(project_root=root))
    response = client.post(
        "/review",
        data={
            "rag_folder": str(root / "examples"),
            "rebuild_index": "true",
            "company": "Example",
            "job_text": "# AI Engineer\nPython SQL RAG retrieval evaluation machine learning",
            "job_url": "",
            "top_k": "8",
        },
    )
    assert response.status_code == 200
    assert "Fit score" in response.text
    assert "CV vs JD Review" in response.text
    assert "CV Improvement Workspace" in response.text
    assert "JD Signals" in response.text
    assert "Evidence used" in response.text
    assert "CV Rewrite Suggestions" in response.text
    assert "data-copy-target" in response.text
    assert "Claim check" in response.text
    assert "Review Diagnostics" in response.text
    assert (root / "outputs" / "latest_brief.json").exists()
    assert list((root / "outputs" / "runs").glob("review-*.json"))


def test_web_review_rejects_empty_jd(tmp_path, monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    root = build_example_root(tmp_path)
    client = TestClient(create_app(project_root=root))
    response = client.post(
        "/review",
        data={
            "rag_folder": str(root / "examples"),
            "rebuild_index": "true",
            "job_text": "",
            "job_url": "",
        },
    )
    assert response.status_code == 400
    assert "Provide exactly one JD input" in response.text


def test_web_review_company_research_disabled_without_key(tmp_path, monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    root = build_example_root(tmp_path)
    client = TestClient(create_app(project_root=root))
    response = client.post(
        "/review",
        data={
            "rag_folder": str(root / "examples"),
            "rebuild_index": "true",
            "company": "Example",
            "job_text": "# Data Scientist\nPython SQL statistics dashboard evaluation",
            "job_url": "",
            "top_k": "8",
        },
    )
    assert response.status_code == 200
    assert "Data Scientist" in response.text


def test_web_review_rejects_rag_folder_outside_project(tmp_path, monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    root = build_example_root(tmp_path)
    outside = tmp_path.parent
    client = TestClient(create_app(project_root=root))
    response = client.post(
        "/review",
        data={
            "rag_folder": str(outside),
            "rebuild_index": "true",
            "company": "Example",
            "job_text": "# Data Scientist\nPython SQL statistics dashboard evaluation",
            "job_url": "",
            "top_k": "8",
        },
    )
    assert response.status_code == 400
    assert "RAG folder must stay inside the project directory" in response.text


def test_web_review_can_save_and_recall_memory(tmp_path, monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    root = build_example_root(tmp_path)
    client = TestClient(create_app(project_root=root))
    response = client.post(
        "/review",
        data={
            "rag_folder": str(root / "examples"),
            "rebuild_index": "true",
            "company": "Example",
            "job_text": "# AI Engineer\nPython SQL RAG retrieval evaluation",
            "job_url": "",
            "memory_note": "I prefer backend AI roles with RAG evaluation.",
            "memory_tags": "preference,career",
            "memory_query": "backend RAG",
            "top_k": "8",
        },
    )
    assert response.status_code == 200
    assert "Memory saved." in response.text
    assert "I prefer backend AI roles" in response.text


def test_web_preview_job_url_populates_editable_jd(tmp_path, monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    root = build_example_root(tmp_path)
    client = TestClient(create_app(project_root=root))

    from career_copilot import web_app

    class Job:
        source_type = "url"
        text = "Previewed JD with Python SQL Docker. Location: Taipei. Preferred: RAG."
        title = "Preview Role"
        company = "Example"
        url = "https://example.com/job"
        cached_path = None
        extraction_method = "json_ld"

    monkeypatch.setattr(web_app, "resolve_job_input", lambda **kwargs: Job())
    response = client.post(
        "/preview-job",
        data={
            "rag_folder": str(root / "examples"),
            "rebuild_index": "true",
            "company": "Example",
            "job_url": "https://example.com/job",
            "job_text": "",
            "top_k": "8",
        },
    )
    assert response.status_code == 200
    assert "JD Preview" in response.text
    assert "json_ld" in response.text
    assert "Previewed JD with Python SQL Docker." in response.text
    assert "Required:" in response.text
    assert "docker" in response.text
    assert "Preferred:" in response.text


def test_web_history_lists_and_loads_review_runs(tmp_path, monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    root = build_example_root(tmp_path)
    client = TestClient(create_app(project_root=root))
    review = client.post(
        "/review",
        data={
            "rag_folder": str(root / "examples"),
            "rebuild_index": "true",
            "company": "Example",
            "job_text": "# AI Engineer\nPython SQL RAG retrieval evaluation",
            "job_url": "",
            "top_k": "8",
        },
    )
    assert review.status_code == 200
    history = client.get("/history")
    assert history.status_code == 200
    assert "Review History" in history.text
    assert "AI Engineer" in history.text
    run_id = next((root / "outputs" / "runs").glob("review-*.json")).name
    detail = client.get(f"/history/{run_id}")
    assert detail.status_code == 200
    assert "Loaded review" in detail.text
    assert "CV vs JD Review" in detail.text


def test_web_eval_dashboard_reports_scoring_and_extraction(tmp_path, monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    root = build_example_root(tmp_path)
    benchmarks = root / "benchmarks"
    benchmarks.mkdir()
    benchmarks.joinpath("scoring_cases.jsonl").write_text(
        '{"id":"strong","job_text":"AI Engineer Python RAG retrieval evaluation systems.",'
        '"resume_docs":["Deployed Python RAG retrieval evaluation systems in production and improved coverage by 30%."],'
        '"expected_score_min":70,"expected_score_max":100,"expected_verdicts":["strong_match","stretch"]}\n',
        encoding="utf-8",
    )
    benchmarks.joinpath("queries.jsonl").write_text(
        '{"query":"What shows RAG retrieval?","relevant_sources":["projects/rag.md"]}\n',
        encoding="utf-8",
    )
    client = TestClient(create_app(project_root=root))
    review = client.post(
        "/review",
        data={
            "rag_folder": str(root / "examples"),
            "rebuild_index": "true",
            "company": "Example",
            "job_text": "# AI Engineer\nPython SQL RAG retrieval evaluation machine learning",
            "job_url": "",
            "top_k": "8",
        },
    )
    assert review.status_code == 200
    response = client.get("/eval")
    assert response.status_code == 200
    assert "Evaluation Dashboard" in response.text
    assert "Scoring Benchmarks" in response.text
    assert "Retrieval Recall@5" in response.text
    assert "JD Extraction" in response.text


def test_web_history_rejects_invalid_run_id(tmp_path, monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    root = build_example_root(tmp_path)
    client = TestClient(create_app(project_root=root))
    response = client.get("/history/../latest_brief.json")
    assert response.status_code == 404
