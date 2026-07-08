import json

from career_copilot.cv_export import export_cv_rewrite, render_cv_rewrite_markdown


def sample_brief():
    return {
        "job_title": "AI Engineer",
        "fit_score": 62,
        "application_verdict": {"label": "stretch"},
        "diagnostics": {"llm_status": "not_configured"},
        "cv_jd_review": {"reason": "Partial match."},
        "cv_rewrite_suggestions": [
            {
                "bullet": "Built work demonstrating RAG.",
                "source_path": "projects/rag.md",
                "chunk_index": 0,
                "target_terms": ["rag"],
                "confidence": 0.7,
                "safe_to_claim": True,
                "claim_verification": {
                    "status": "supported",
                    "reason": "The target terms appear in the cited evidence.",
                    "supported_claims": ["rag"],
                    "risky_claims": [],
                    "required_evidence": [],
                },
            }
        ],
        "recommended_actions": ["Rewrite the CV."],
    }


def test_render_cv_rewrite_markdown_contains_safe_claim_flag():
    text = render_cv_rewrite_markdown(sample_brief())
    assert "# CV Rewrite Plan - AI Engineer" in text
    assert "Safe to claim now: yes" in text
    assert "Claim check: supported" in text
    assert "`projects/rag.md`" in text


def test_export_cv_rewrite_writes_markdown(tmp_path):
    brief = tmp_path / "brief.json"
    output = tmp_path / "rewrite.md"
    brief.write_text(json.dumps(sample_brief()), encoding="utf-8")
    export_cv_rewrite(brief, output)
    assert output.exists()
    assert "Built work demonstrating RAG" in output.read_text(encoding="utf-8")
