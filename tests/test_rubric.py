from career_copilot.rubric import (
    analyze_evidence_depth,
    analyze_job_requirements,
    score_with_rubric,
)


def hit(text: str, category: str = "resume") -> dict:
    return {
        "text": text,
        "category": category,
        "filename": f"{category}.md",
        "source_path": f"{category}/{category}.md",
        "chunk_index": 0,
        "score": 0.8,
    }


def test_role_detection_for_data_scientist_terms():
    requirements = analyze_job_requirements(
        "Data Scientist role using statistics, SQL, experiments, dashboards, pandas, and forecasting."
    )
    assert requirements["role_family"] == "data_scientist"
    assert {"statistics", "sql", "ab_test", "dashboard", "pandas", "forecasting"} <= set(requirements["required"])


def test_role_detection_for_ai_engineer_terms():
    requirements = analyze_job_requirements("AI Engineer building RAG, LLM agents, embeddings, and vector search.")
    assert requirements["role_family"] == "ai_engineer"
    assert {"rag", "llm", "agent", "embedding", "vector_search"} <= set(requirements["required"])


def test_ai_engineer_downweights_contextual_prompting_and_dashboards():
    requirements = analyze_job_requirements(
        """
        AI Engineer
        Responsibilities:
        Build RAG and LLM retrieval evaluation systems.

        About the team:
        Internal dashboards track usage, and some teams experiment with prompting.
        """
    )
    assert requirements["role_family"] == "ai_engineer"
    assert {"rag", "llm", "retrieval", "evaluation"} <= set(requirements["required"])
    assert "dashboard" not in requirements["required"]
    assert "prompting" not in requirements["required"]
    assert {"dashboard", "prompting"} & set(requirements["ignored"])


def test_data_scientist_treats_dashboard_as_role_relevant():
    requirements = analyze_job_requirements(
        """
        Data Scientist
        Required Qualifications:
        SQL, statistics, dashboards, forecasting, and A/B testing.
        """
    )
    assert requirements["role_family"] == "data_scientist"
    assert {"sql", "statistics", "dashboard", "forecasting", "ab_test"} <= set(requirements["required"])


def test_preferred_section_does_not_become_required():
    requirements = analyze_job_requirements(
        """
        AI Engineer
        Required Qualifications:
        Python, RAG, and retrieval evaluation.

        Preferred Qualifications:
        Prompt engineering and embeddings.
        """
    )
    assert {"python", "rag", "retrieval", "evaluation"} <= set(requirements["required"])
    assert {"prompting", "embedding"} <= set(requirements["preferred"])
    assert "prompting" not in requirements["required"]


def test_shallow_mentions_do_not_score_as_strong_match():
    requirements = analyze_job_requirements("AI Engineer using Python, RAG, retrieval, and evaluation.")
    depth = analyze_evidence_depth(
        requirements,
        [hit("Python RAG retrieval evaluation.")],
        [],
    )
    score = score_with_rubric(requirements, depth, [hit("Python RAG retrieval evaluation.")])
    assert score["fit_score"] < 75
    assert set(score["weak_evidence"]) >= {"python", "rag", "retrieval", "evaluation"}


def test_work_impact_in_cv_increases_rubric_score():
    requirements = analyze_job_requirements("AI Engineer deploys Python RAG retrieval evaluation systems.")
    resume_hits = [
        hit(
            "Deployed Python RAG retrieval evaluation systems in production and improved benchmark coverage by 30%."
        )
    ]
    depth = analyze_evidence_depth(requirements, resume_hits, [])
    score = score_with_rubric(requirements, depth, resume_hits)
    assert score["fit_score"] >= 75
    assert score["scoring_breakdown"]["quantified_impact"] == 10


def test_generic_impact_words_without_metrics_do_not_get_quantified_impact():
    requirements = analyze_job_requirements("AI Engineer deploys Python RAG retrieval evaluation systems.")
    resume_hits = [
        hit("Improved and automated Python RAG retrieval evaluation workflows.", category="resume")
    ]
    depth = analyze_evidence_depth(requirements, resume_hits, [])
    score = score_with_rubric(requirements, depth, resume_hits)
    assert score["scoring_breakdown"]["quantified_impact"] == 0
    assert score["fit_score"] < 75
