from career_copilot.contracts import validate_brief


def test_validate_brief_accepts_valid_payload():
    payload = {
        "job_title": "AI Engineer",
        "fit_score": 80,
        "role_family": "ai_engineer",
        "jd_requirements": {"required": ["python"], "preferred": [], "responsibilities": []},
        "evidence_depth": {"by_term": {}},
        "scoring_breakdown": {
            "required_skill_coverage": 45,
            "responsibility_alignment": 20,
            "evidence_depth": 10,
            "quantified_impact": 5,
            "preferred_coverage": 0,
        },
        "weak_evidence": [],
        "cv_jd_review": {
            "score": 80,
            "verdict": "strong_cv_match",
            "reason": "Strong evidence.",
            "matched_terms": ["python"],
            "missing_from_cv": [],
            "weak_evidence": [],
            "scoring_breakdown": {},
        },
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
