from pathlib import Path

from career_copilot.answer import build_citations
from career_copilot.documents import clean_text, chunk_document


def test_clean_text_normalizes_whitespace():
    assert clean_text("hello   world\r\n\r\n\r\nnext") == "hello world\n\nnext"


def test_chunk_document_tracks_paragraphs(tmp_path: Path):
    folder = tmp_path / "projects"
    folder.mkdir()
    path = folder / "note.md"
    path.write_text("First paragraph.\n\nSecond paragraph has more text.", encoding="utf-8")
    chunks = chunk_document(path, path.read_text(encoding="utf-8"), root=tmp_path, chunk_size=40, overlap=5)
    assert chunks
    assert chunks[0].filename == "note.md"
    assert chunks[0].category == "projects"
    assert chunks[0].paragraph_start == 1


def test_chunk_document_tracks_pdf_page_markers(tmp_path: Path):
    folder = tmp_path / "resume"
    folder.mkdir()
    path = folder / "cv.pdf"
    text = "[PDF page 1]\nPython SQL.\n\n[PDF page 2]\nDocker Kubernetes deployment."
    chunks = chunk_document(path, text, root=tmp_path, chunk_size=45, overlap=5)
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 2
    hit = chunks[-1].__dict__ | {"score": 0.9}
    citation = build_citations([hit])[0]
    assert citation["pages"][-1] == 2
