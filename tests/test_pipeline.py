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
