from pathlib import Path

import pytest

from career_copilot.job_input import (
    infer_company_from_url,
    resolve_job_input,
    sanitize_filename,
)


def test_resolve_job_input_from_file(tmp_path: Path):
    job = tmp_path / "job.md"
    job.write_text("# Data Scientist\nPython SQL experimentation", encoding="utf-8")
    result = resolve_job_input(job_file=job)
    assert result.source_type == "file"
    assert result.title == "Data Scientist"
    assert "Python SQL" in result.text


def test_resolve_job_input_from_text_and_cache(tmp_path: Path):
    result = resolve_job_input(
        job_text="# AI Engineer\nBuild RAG systems",
        cache_dir=tmp_path,
    )
    assert result.source_type == "text"
    assert result.title == "AI Engineer"
    assert result.cached_path is not None
    assert Path(result.cached_path).exists()


def test_resolve_job_input_from_stdin_text():
    result = resolve_job_input(use_stdin=True, stdin_text="# ML Engineer\nEvaluate models")
    assert result.source_type == "stdin"
    assert result.title == "ML Engineer"


def test_resolve_job_input_rejects_multiple_inputs(tmp_path: Path):
    job = tmp_path / "job.md"
    job.write_text("job", encoding="utf-8")
    with pytest.raises(ValueError):
        resolve_job_input(job_file=job, job_text="also job")


def test_infer_company_from_url():
    assert infer_company_from_url("https://careers.nvidia.com/jobs/123") == "Nvidia"
    assert infer_company_from_url("https://www.example-company.com/careers") == "Example Company"


def test_sanitize_filename():
    assert sanitize_filename("AI Engineer / RAG Role!") == "ai-engineer-rag-role"
