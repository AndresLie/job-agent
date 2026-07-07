from career_copilot.contracts import validate_brief


def test_validate_brief_accepts_valid_payload():
    payload = {
        "job_title": "AI Engineer",
        "fit_score": 80,
        "matched_evidence": [],
        "application_verdict": {
            "label": "strong_match",
            "apply_now": True,
            "risk_level": "low",
            "reason": "Strong evidence.",
        },
        "cv_rewrite_suggestions": [],
        "skill_gaps": [],
        "recommended_actions": [],
        "citations": [],
        "confidence": 0.8,
    }
    assert validate_brief(payload) == (True, "")


def test_validate_brief_rejects_missing_field():
    ok, reason = validate_brief({"job_title": "AI Engineer"})
    assert not ok
    assert "missing fields" in reason
