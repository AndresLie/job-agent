from pathlib import Path

from career_copilot.answer import answer_query
from career_copilot.brief import generate_brief, skill_terms, verify_rewrite_claim
from career_copilot.documents import chunk_document
from career_copilot.embeddings import HashingEmbedder
from career_copilot.hermes import detect_agent_contradictions, generate_brutal_llm_review, parse_agent_output
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
    assert brief["schema_version"] == "1.1"
    assert brief["matched_evidence"]
    assert brief["score_explanations"]
    assert "llm_agent_steps" in brief
    assert "agent_trace" in brief
    assert brief["agent_consensus"]["source"] == "deterministic_fallback"
    assert brief["diagnostics"]["llm_status"] in {"not_configured", "configured_but_unavailable", "used"}


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
    assert brief["cv_jd_review"]["score"] == brief["fit_score"]


def test_generate_brief_uses_requirement_override(tmp_path, monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    store, embedder, memory = build_store(tmp_path)
    brief = generate_brief(
        None,
        store,
        embedder,
        memory,
        job_text="# AI Engineer\nPython RAG retrieval evaluation",
        job_title="AI Engineer",
        requirements_override={
            "role_family": "software_engineer",
            "required": "retrieval",
            "preferred": "python",
            "ignored": "rag, evaluation",
        },
    )
    assert brief["role_family"] == "software_engineer"
    assert brief["jd_requirements"]["required"] == ["retrieval"]
    assert brief["jd_requirements"]["preferred"] == ["python"]
    assert set(brief["jd_requirements"]["ignored"]) == {"evaluation", "rag"}
    assert brief["jd_requirements"]["override_applied"] is True


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


def test_brief_scores_each_resume_file_separately(tmp_path, monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    embedder = HashingEmbedder()
    store = JsonVectorStore(tmp_path / "vectors.json")
    memory = MemoryStore(tmp_path / "memory.json")
    resume_texts = {
        "resume/python_cv.md": "Python RAG retrieval evaluation and LLM agent coursework.",
        "resume/platform_cv.md": "Docker Kubernetes APIs production deployment improved latency by 30%.",
    }
    for relpath, text in resume_texts.items():
        path = tmp_path / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        store.upsert_chunks(chunk_document(path, text, root=tmp_path), embedder)

    brief = generate_brief(
        None,
        store,
        embedder,
        memory,
        job_text="# AI Platform Engineer\nPython RAG retrieval evaluation Docker Kubernetes APIs",
        job_title="AI Platform Engineer",
    )

    selected = next(item for item in brief["cv_rankings"] if item["selected"])
    unselected = next(item for item in brief["cv_rankings"] if not item["selected"])
    active_terms = skill_terms(resume_texts[selected["source_path"]])
    inactive_terms = skill_terms(resume_texts[unselected["source_path"]])

    assert len(brief["cv_rankings"]) == 2
    assert brief["active_resume"]["source_path"] == selected["source_path"]
    assert brief["fit_score"] == selected["fit_score"]
    assert all(
        item["source_path"] == selected["source_path"]
        for item in brief["matched_evidence"]
        if item.get("source") != "memory"
    )
    assert set(brief["cv_match"]["matched_terms"]) <= active_terms
    assert not (set(brief["cv_match"]["matched_terms"]) >= (active_terms | inactive_terms))


def test_web_research_does_not_change_cv_jd_score(tmp_path, monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    embedder = HashingEmbedder()
    store = JsonVectorStore(tmp_path / "vectors.json")
    memory = MemoryStore(tmp_path / "memory.json")
    resume = tmp_path / "resume" / "cv.md"
    resume.parent.mkdir()
    resume.write_text("Python SQL analytics.", encoding="utf-8")
    store.upsert_chunks(chunk_document(resume, resume.read_text(encoding="utf-8"), root=tmp_path), embedder)

    base = generate_brief(
        None,
        store,
        embedder,
        memory,
        job_text="# Data Scientist\nPython SQL",
        job_title="Data Scientist",
    )
    with_research = generate_brief(
        None,
        store,
        embedder,
        memory,
        job_text="# Data Scientist\nPython SQL",
        job_title="Data Scientist",
        web_research=[
            {
                "title": "Company AI Platform",
                "url": "https://example.com",
                "source_type": "exa",
                "highlights": ["Docker Kubernetes RAG LLM vector search"],
                "summary": "The company uses many AI platform technologies.",
            }
        ],
    )

    assert with_research["fit_score"] == base["fit_score"]
    assert with_research["cv_jd_review"]["matched_terms"] == base["cv_jd_review"]["matched_terms"]


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
    assert suggestion["source_category"] == "experience"
    assert "machine learning evaluation pipelines" in suggestion["evidence_excerpt"]
    assert suggestion["safe_to_claim"] is True
    assert suggestion["claim_verification"]["status"] == "supported"
    assert isinstance(suggestion["chunk_index"], int)
    assert "Quantify impact if true." in suggestion["bullet"]


def test_claim_verifier_flags_unsupported_impact_language():
    result = verify_rewrite_claim(
        {"text": "Built a RAG retrieval evaluation prototype."},
        ["rag", "retrieval", "evaluation"],
        "Deployed RAG retrieval evaluation to production and improved latency by 30%.",
    )
    assert result["status"] == "needs_verification"
    assert "deployment evidence" in result["required_evidence"]


def test_hermes_review_runs_multi_step_agent(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    calls = []

    def fake_nvidia_chat(system_prompt, user_prompt):
        calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        return f"step {len(calls)} conclusion"

    monkeypatch.setattr("career_copilot.hermes.nvidia_chat", fake_nvidia_chat)
    review = generate_brutal_llm_review(
        job_title="AI Engineer",
        job_text="Python retrieval evaluation Docker",
        fit_score=56,
        application_verdict={"label": "stretch", "reason": "Some CV evidence is missing."},
        cv_rewrite_suggestions=[
            {
                "bullet": "Built retrieval evaluation with Docker. Quantify impact if true.",
                "source_path": "projects/rag.md",
                "chunk_index": 0,
            }
        ],
        requirements={"role_family": "ai_engineer", "required": ["python", "retrieval", "docker"], "preferred": []},
        evidence_depth={"python": {"resume": 1}, "docker": {"supporting": 1}},
        scoring_breakdown={"base": 56},
        weak_evidence=["docker"],
        matched=["python"],
        hidden_terms=["docker", "retrieval"],
        gaps=["kubernetes"],
        resume_hits=[
            {
                "text": "Python data analysis.",
                "category": "resume",
                "filename": "cv.md",
                "chunk_index": 0,
            }
        ],
        supporting_hits=[
            {
                "text": "Built a RAG system with retrieval evaluation and Docker deployment.",
                "category": "projects",
                "filename": "rag.md",
                "chunk_index": 0,
            }
        ],
    )

    assert review is not None
    assert [step["id"] for step in review["steps"]] == [
        "jd_analyst",
        "cv_match",
        "evidence_miner",
        "claim_verifier",
        "critic",
        "contradiction_judge",
        "final_synthesizer",
    ]
    assert [item["agent_id"] for item in review["agent_trace"]] == [
        "jd_analyst",
        "cv_match",
        "evidence_miner",
        "claim_verifier",
        "critic",
        "contradiction_judge",
        "final_synthesizer",
    ]
    assert "Hermes multi-step review" in review["final_review"]
    assert "Prior agent conclusions:\nNone yet." in calls[0]["user_prompt"]
    assert "step 1 conclusion" in calls[1]["user_prompt"]
    assert "Project/experience evidence:" not in calls[1]["user_prompt"]
    assert "Built a RAG system" not in calls[1]["user_prompt"]
    assert review["agent_consensus"]["source"] == "hermes_multi_agent"
    assert review["agent_consensus"]["successful_agents"] == 7
    assert review["agent_trace"][0]["structured_output"]["conclusion"] == "step 1 conclusion"
    assert review["agent_trace"][0]["structured"] is False
    assert review["agent_trace"][0]["schema_errors"] == ["invalid_json"]
    assert review["agent_trace"][0]["usage"]["latency_ms"] >= 0
    assert review["agent_consensus"]["total_prompt_chars"] > 0


def test_hermes_review_captures_failed_agent(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    calls = []

    def fake_nvidia_chat(system_prompt, user_prompt):
        calls.append(user_prompt)
        if len(calls) == 3:
            return None
        return f"agent {len(calls)} ok"

    monkeypatch.setattr("career_copilot.hermes.nvidia_chat", fake_nvidia_chat)
    review = generate_brutal_llm_review(
        job_title="AI Engineer",
        job_text="Python retrieval evaluation Docker",
        fit_score=56,
        application_verdict={"label": "stretch", "reason": "Some CV evidence is missing."},
        cv_rewrite_suggestions=[],
        requirements={"role_family": "ai_engineer", "required": ["python", "docker"], "preferred": []},
        evidence_depth={},
        scoring_breakdown={"base": 56},
        weak_evidence=["docker"],
        matched=["python"],
        hidden_terms=["docker"],
        gaps=["kubernetes"],
        resume_hits=[],
        supporting_hits=[],
    )

    assert review is not None
    assert len(review["agent_trace"]) == 7
    assert review["agent_trace"][2]["status"] == "failed"
    assert review["agent_trace"][2]["failure_reason"]
    assert len(review["steps"]) == 6
    assert "failed" in calls[3]


def test_hermes_review_parses_structured_agent_output_and_detects_contradiction(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")

    def fake_nvidia_chat(system_prompt, user_prompt):
        return (
            '{"conclusion":"Strong match apply now","findings":["Strong match"],'
            '"risks":["none"],"recommendations":["apply immediately"],"confidence":0.9}'
        )

    monkeypatch.setattr("career_copilot.hermes.nvidia_chat", fake_nvidia_chat)
    review = generate_brutal_llm_review(
        job_title="AI Engineer",
        job_text="Python retrieval evaluation Docker",
        fit_score=25,
        application_verdict={"label": "weak_match", "apply_now": False, "reason": "Too weak."},
        cv_rewrite_suggestions=[],
        requirements={"role_family": "ai_engineer", "required": ["python", "docker"], "preferred": []},
        evidence_depth={},
        scoring_breakdown={"base": 25},
        weak_evidence=["docker"],
        matched=["python"],
        hidden_terms=[],
        gaps=["kubernetes"],
        resume_hits=[],
        supporting_hits=[],
    )

    assert review is not None
    assert review["agent_trace"][0]["structured"] is True
    assert review["agent_trace"][0]["structured_output"]["confidence"] == 0.9
    assert review["agent_contradictions"]
    assert review["agent_consensus"]["contradiction_count"] >= 1


def test_hermes_structured_output_requires_raw_schema_types():
    payload, structured, errors = parse_agent_output(
        '{"conclusion":"Useful audit","findings":"not a list","risks":[],"recommendations":[],"confidence":0.8}'
    )

    assert payload["findings"] == ["not a list"]
    assert structured is False
    assert "invalid_findings" in errors


def test_contradiction_detector_ignores_negated_recommendations():
    contradictions = detect_agent_contradictions(
        trace=[
            {
                "agent_id": "claim_verifier",
                "status": "success",
                "output": "Docker claim needs verification.",
            },
            {
                "agent_id": "final_synthesizer",
                "status": "success",
                "output": "This is not a strong match. Do not apply immediately. Docker is not safe to claim yet.",
            },
        ],
        fit_score=25,
        application_verdict={"apply_now": False},
        gaps=["docker"],
    )

    assert contradictions == []
