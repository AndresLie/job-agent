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


def test_alias_matching_uses_term_boundaries():
    requirements = analyze_job_requirements(
        "Company follows labor laws and builds storage products for memory systems."
    )
    assert "aws" not in requirements["required"]
    assert "rag" not in requirements["required"]


def test_micron_mes_jd_extracts_manufacturing_it_requirements():
    requirements = analyze_job_requirements(
        """
        IT Software Engineer - MES systems
        Responsibilities:
        Develop, maintain, troubleshoot and support software systems for manufacturing semiconductors.
        Provide incident and problem management for MES systems.
        Participate in testing of new software releases with all sites.
        Track and report indices to monitor systems performance.

        Requirements:
        Database technologies such as Microsoft SQL, Oracle or similar products.
        Linux, Unix and Windows environment.
        Good knowledge in C#/C++ programming and Perl scripting, with working knowledge of SQL or PL/SQL.
        Knowledge in MES systems, and understanding of semiconductor manufacturing process will be an advantage.
        """
    )
    assert requirements["role_family"] == "manufacturing_it"
    assert {
        "csharp",
        "cpp",
        "perl",
        "sql",
        "oracle",
        "pl_sql",
        "linux",
        "unix",
        "windows",
        "mes",
        "incident_management",
        "production_support",
        "troubleshooting",
    } <= set(requirements["required"])
    assert "aws" not in requirements["required"]
    assert "rag" not in requirements["required"]


def test_full_stack_software_jd_is_not_ai_engineer_from_department_label():
    requirements = analyze_job_requirements(
        """
        SOFTWARE DEVELOPMENT ENGINEER
        Department: Smart MFG/AI
        Qualifications and Skills:
        Developing restful web services using NodeJS, Apache or C#
        Object Oriented Programming preferably .NET
        SQL and NoSQL databases
        Angular, TypeScript, JavaScript, and other Web technologies
        Git
        Docker, Kubernetes, OpenShift, or other container technologies
        GitHub Copilot or other Code Generation Utilities
        """
    )

    assert requirements["role_family"] == "software_engineer"
    assert {
        "nodejs",
        "apache",
        "csharp",
        "dotnet",
        "angular",
        "typescript",
        "javascript",
        "git",
        "docker",
        "kubernetes",
        "openshift",
        "code_generation_tools",
    } <= set(requirements["required"])
    assert any({"nodejs", "apache", "csharp"} <= set(group["terms"]) for group in requirements["requirement_groups"])
    assert any({"docker", "kubernetes", "openshift"} <= set(group["terms"]) for group in requirements["requirement_groups"])


def test_core_stack_gap_caps_infrastructure_only_cv():
    requirements = analyze_job_requirements(
        """
        Software Development Engineer
        Requirements:
        Developing restful web services using NodeJS, Apache or C#
        Object Oriented Programming preferably .NET
        SQL and NoSQL databases
        Angular, TypeScript, JavaScript, and other Web technologies
        Git
        Docker, Kubernetes, OpenShift, or other container technologies
        """
    )
    resume_hits = [hit("Built SQL platforms with Docker and Kubernetes and improved throughput by 30%.")]
    depth = analyze_evidence_depth(requirements, resume_hits, [])
    score = score_with_rubric(requirements, depth, resume_hits)

    assert score["fit_score"] <= 45
    assert score["scoring_breakdown"]["score_cap"] <= 45
    assert "dotnet" in score["skill_gaps"]
    assert "angular" in score["skill_gaps"]


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
