from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".pdf"}


@dataclass(frozen=True)
class Chunk:
    id: str
    source_path: str
    category: str
    filename: str
    chunk_index: int
    text: str
    paragraph_start: int
    paragraph_end: int
    page_start: int | None = None
    page_end: int | None = None


def discover_documents(source: Path) -> list[Path]:
    source = source.resolve()
    if source.is_file():
        return [source] if source.suffix.lower() in SUPPORTED_EXTENSIONS else []
    if not source.exists():
        return []
    return [
        path
        for path in sorted(source.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def load_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return load_pdf(path)
    return path.read_text(encoding="utf-8", errors="ignore")


def load_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Install pypdf to ingest PDF files.") from exc
    reader = PdfReader(str(path))
    return "\n\n".join(
        f"[PDF page {index}]\n{page.extract_text() or ''}"
        for index, page in enumerate(reader.pages, start=1)
    )


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def paragraph_spans(text: str) -> list[tuple[int, int, int]]:
    spans: list[tuple[int, int, int]] = []
    for index, match in enumerate(
        re.finditer(r"\S[\s\S]*?(?=(?:\n\s*\n)+|\Z)", text),
        start=1,
    ):
        spans.append((match.start(), match.end(), index))
    return spans or ([(0, len(text), 1)] if text.strip() else [])


def paragraph_range(
    spans: list[tuple[int, int, int]],
    start: int,
    end: int,
) -> tuple[int, int]:
    hits = [number for span_start, span_end, number in spans if span_end > start and span_start < end]
    return (hits[0], hits[-1]) if hits else (1, 1)


def page_spans(text: str) -> list[tuple[int, int, int]]:
    markers = [(match.start(), int(match.group(1))) for match in re.finditer(r"\[PDF page (\d+)\]", text)]
    spans = []
    for index, (start, page_number) in enumerate(markers):
        end = markers[index + 1][0] if index + 1 < len(markers) else len(text)
        spans.append((start, end, page_number))
    return spans


def page_range(spans: list[tuple[int, int, int]], start: int, end: int) -> tuple[int | None, int | None]:
    hits = [number for span_start, span_end, number in spans if span_end > start and span_start < end]
    return (hits[0], hits[-1]) if hits else (None, None)


def chunk_document(
    path: Path,
    text: str,
    root: Path,
    chunk_size: int = 900,
    overlap: int = 150,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")
    text = clean_text(text)
    if not text:
        return []

    spans = paragraph_spans(text)
    pages = page_spans(text)
    chunks: list[Chunk] = []
    cursor = 0
    step = chunk_size - overlap
    while cursor < len(text):
        end = find_chunk_end(text, cursor, min(cursor + chunk_size, len(text)))
        value = text[cursor:end].strip()
        if value:
            p_start, p_end = paragraph_range(spans, cursor, end)
            page_start, page_end = page_range(pages, cursor, end)
            relpath = path.resolve().relative_to(root.resolve()).as_posix()
            category = infer_category(relpath)
            index = len(chunks)
            chunks.append(
                Chunk(
                    id=f"{relpath}:{index}",
                    source_path=relpath,
                    category=category,
                    filename=path.name,
                    chunk_index=index,
                    text=value,
                    paragraph_start=p_start,
                    paragraph_end=p_end,
                    page_start=page_start,
                    page_end=page_end,
                )
            )
        if end >= len(text):
            break
        cursor = max(end - overlap, cursor + step)
    return chunks


def infer_category(relpath: str) -> str:
    first = relpath.split("/", 1)[0].casefold()
    if first in {"projects", "project"}:
        return "projects"
    if first in {"experience", "experiences", "work"}:
        return "experience"
    if first in {"resume", "cv", "resumes"}:
        return "resume"
    if first in {"jobs", "job", "job_postings"}:
        return "jobs"
    return "general"


def find_chunk_end(text: str, start: int, tentative_end: int) -> int:
    if tentative_end >= len(text):
        return len(text)
    for marker, window in (("\n\n", 140), (". ", 100), ("! ", 100), ("? ", 100), ("\n", 80)):
        boundary = text.find(marker, tentative_end, min(tentative_end + window, len(text)))
        if boundary != -1:
            return boundary + len(marker)
    whitespace = text.rfind(" ", start, tentative_end)
    return whitespace if whitespace > start else tentative_end
