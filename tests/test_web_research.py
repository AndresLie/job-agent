import pytest

from career_copilot.web_research import normalize_exa_results, research_company


def test_normalize_exa_results_from_dicts():
    response = [
        {
            "title": "Example Corp Engineering",
            "url": "https://example.com/engineering",
            "highlights": ["Builds AI systems", "Uses Python"],
            "summary": "Engineering overview",
        }
    ]
    sources = normalize_exa_results(response)
    assert sources[0].title == "Example Corp Engineering"
    assert sources[0].source_type == "exa"
    assert sources[0].highlights == ["Builds AI systems", "Uses Python"]


def test_research_company_requires_key(monkeypatch):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="EXA_API_KEY"):
        research_company(company="Example", role="AI Engineer")
