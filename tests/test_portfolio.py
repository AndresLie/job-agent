from career_copilot.portfolio import build_portfolio_report, export_portfolio_report


def test_build_portfolio_report_scores_role_coverage(tmp_path):
    project = tmp_path / "projects" / "rag.md"
    project.parent.mkdir()
    project.write_text("Built Python RAG LLM retrieval evaluation with vector search.", encoding="utf-8")
    experience = tmp_path / "experience" / "platform.md"
    experience.parent.mkdir()
    experience.write_text("Deployed Docker Kubernetes APIs for machine learning.", encoding="utf-8")
    report = build_portfolio_report(tmp_path)
    assert report["documents"] == 2
    assert report["roles"]["ai_engineer"]["coverage"] > 0
    assert report["roles"]["ml_engineer"]["strongest_sources"]


def test_export_portfolio_report_writes_markdown(tmp_path):
    project = tmp_path / "projects" / "rag.md"
    project.parent.mkdir()
    project.write_text("Python RAG LLM retrieval evaluation.", encoding="utf-8")
    output = tmp_path / "portfolio.md"
    export_portfolio_report(tmp_path, output)
    text = output.read_text(encoding="utf-8")
    assert "# Portfolio Report" in text
    assert "Ai Engineer" in text
