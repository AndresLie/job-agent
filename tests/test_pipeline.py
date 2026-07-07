from pathlib import Path

from career_copilot.answer import answer_query
from career_copilot.brief import generate_brief
from career_copilot.documents import chunk_document
from career_copilot.embeddings import HashingEmbedder
from career_copilot.memory import MemoryStore
from career_copilot.vector_store import JsonVectorStore


def build_store(tmp_path: Path):
    project_dir = tmp_path / "projects"
    project_dir.mkdir()
    source = project_dir / "projects.md"
    source.write_text(
        "RAG project used retrieval augmented generation and citations.\n\n"
        "Memory project used BM25 to recall durable preferences.",
        encoding="utf-8",
    )
    embedder = HashingEmbedder()
    store = JsonVectorStore(tmp_path / "vectors.json")
    store.upsert_chunks(chunk_document(source, source.read_text(encoding="utf-8"), root=tmp_path), embedder)
    memory = MemoryStore(tmp_path / "memory.json")
    memory.add("Candidate prefers AI engineering roles with retrieval evaluation.")
    return store, embedder, memory


def test_answer_query_returns_citations(tmp_path):
    store, embedder, memory = build_store(tmp_path)
    result = answer_query("What project shows retrieval?", store, embedder, memory, use_llm=False)
    assert result["citations"]
    assert "Grounded answer" in result["answer"]


def test_generate_brief_has_required_fields(tmp_path):
    store, embedder, memory = build_store(tmp_path)
    job = tmp_path / "job.md"
    job.write_text("# AI Engineer\nPython retrieval evaluation BM25", encoding="utf-8")
    brief = generate_brief(job, store, embedder, memory)
    assert brief["job_title"] == "AI Engineer"
    assert brief["matched_evidence"]


def test_generate_brief_from_text_with_web_research(tmp_path):
    store, embedder, memory = build_store(tmp_path)
    brief = generate_brief(
        None,
        store,
        embedder,
        memory,
        job_text="# AI Engineer\nPython retrieval evaluation",
        job_title="AI Engineer",
        job_company="Example",
        job_source_type="text",
        web_research=[
            {
                "title": "Example Engineering",
                "url": "https://example.com",
                "source_type": "exa",
                "highlights": ["Example builds AI platforms"],
                "summary": "Company engineering context",
            }
        ],
    )
    assert brief["job_input"]["source_type"] == "text"
    assert brief["job_input"]["company"] == "Example"
    assert brief["web_research"][0]["source_type"] == "exa"


def test_brief_scores_resume_only_and_uses_projects_for_cv_recommendations(tmp_path, monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    embedder = HashingEmbedder()
    store = JsonVectorStore(tmp_path / "vectors.json")
    memory = MemoryStore(tmp_path / "memory.json")

    for folder, filename, text in [
        ("resume", "cv.md", "Python data analysis and SQL reporting."),
        ("projects", "rag.md", "Built a RAG retrieval evaluation project with citations."),
        ("jobs", "cached_job.md", "Python RAG retrieval evaluation machine learning agent."),
    ]:
        path = tmp_path / folder / filename
        path.parent.mkdir(exist_ok=True)
        path.write_text(text, encoding="utf-8")
        store.upsert_chunks(chunk_document(path, text, root=tmp_path), embedder)

    brief = generate_brief(
        None,
        store,
        embedder,
        memory,
        job_text="# AI Engineer\nPython RAG retrieval evaluation",
        job_title="AI Engineer",
    )

    assert brief["fit_score"] < 100
    assert brief["cv_match"]["matched_terms"] == ["python"]
    assert any("rag" in item["terms_to_add_to_cv"] for item in brief["hidden_evidence"])
    assert all(item["category"] != "jobs" for item in brief["citations"])
    assert brief["application_verdict"]["label"] in {"weak_match", "stretch"}
    assert brief["cv_rewrite_suggestions"]
    assert all("jobs/" not in item["source_path"] for item in brief["cv_rewrite_suggestions"])


def test_brief_verdict_not_competitive_without_resume(tmp_path, monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    store, embedder, memory = build_store(tmp_path)

    brief = generate_brief(
        None,
        store,
        embedder,
        memory,
        job_text="# AI Engineer\nPython RAG retrieval evaluation",
        job_title="AI Engineer",
    )

    assert brief["fit_score"] == 0
    assert brief["application_verdict"]["label"] == "not_competitive"
    assert brief["application_verdict"]["apply_now"] is False


def test_brief_verdict_strong_match_for_high_resume_overlap(tmp_path, monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    embedder = HashingEmbedder()
    store = JsonVectorStore(tmp_path / "vectors.json")
    memory = MemoryStore(tmp_path / "memory.json")
    resume = tmp_path / "resume" / "cv.md"
    resume.parent.mkdir()
    resume.write_text(
        "Deployed Python SQL RAG retrieval evaluation machine learning LLM agent work "
        "that improved benchmark coverage by 30%.",
        encoding="utf-8",
    )
    store.upsert_chunks(chunk_document(resume, resume.read_text(encoding="utf-8"), root=tmp_path), embedder)

    brief = generate_brief(
        None,
        store,
        embedder,
        memory,
        job_text="# AI Engineer\nPython SQL RAG retrieval evaluation",
        job_title="AI Engineer",
    )

    assert brief["fit_score"] >= 75
    assert brief["application_verdict"]["label"] == "strong_match"
    assert brief["application_verdict"]["apply_now"] is True


def test_cv_rewrite_suggestions_are_grounded_in_supporting_evidence(tmp_path, monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    embedder = HashingEmbedder()
    store = JsonVectorStore(tmp_path / "vectors.json")
    memory = MemoryStore(tmp_path / "memory.json")

    for folder, filename, text in [
        ("resume", "cv.md", "Python SQL analytics."),
        ("experience", "internship.md", "Delivered machine learning evaluation pipelines with RAG retrieval checks."),
        ("jobs", "job.md", "RAG retrieval evaluation agent machine learning."),
    ]:
        path = tmp_path / folder / filename
        path.parent.mkdir(exist_ok=True)
        path.write_text(text, encoding="utf-8")
        store.upsert_chunks(chunk_document(path, text, root=tmp_path), embedder)

    brief = generate_brief(
        None,
        store,
        embedder,
        memory,
        job_text="# Data Scientist\nPython machine learning RAG retrieval evaluation",
        job_title="Data Scientist",
    )

    assert brief["cv_rewrite_suggestions"]
    suggestion = brief["cv_rewrite_suggestions"][0]
    assert suggestion["source_path"].startswith("experience/")
    assert isinstance(suggestion["chunk_index"], int)
    assert "Quantify impact if true." in suggestion["bullet"]
