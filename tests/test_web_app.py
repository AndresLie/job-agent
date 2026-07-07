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
            "job_text": "# AI Engineer\nPython SQL RAG retrieval evaluation",
            "job_url": "",
            "top_k": "8",
        },
    )
    assert response.status_code == 200
    assert "Fit score" in response.text
    assert "CV vs JD Review" in response.text
    assert "CV Rewrite Suggestions" in response.text
    assert (root / "outputs" / "latest_brief.json").exists()


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
