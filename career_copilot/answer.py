from __future__ import annotations

import os

from .memory import MemoryStore
from .vector_store import JsonVectorStore
from .embeddings import Embedder


def answer_query(
    query: str,
    store: JsonVectorStore,
    embedder: Embedder,
    memory: MemoryStore,
    top_k: int = 5,
    use_llm: bool = True,
) -> dict:
    hits = store.query(query, embedder, top_k)
    memories = memory.retrieve(query, k=3)
    citations = build_citations(hits)
    answer = None
    used_llm = False
    if use_llm and has_llm_config():
        answer = llm_answer(query, hits, [item.summary for item in memories])
        used_llm = answer is not None
    if answer is None:
        answer = extractive_answer(query, hits, [item.summary for item in memories])
    return {
        "query": query,
        "answer": answer,
        "citations": citations,
        "retrieved_chunks": hits,
        "used_llm": used_llm,
    }


def build_citations(hits: list[dict]) -> list[dict]:
    return [
        {
            "source": hit["filename"],
            "chunk_index": hit["chunk_index"],
            "paragraphs": [hit["paragraph_start"], hit["paragraph_end"]],
            "score": round(hit["score"], 4),
        }
        for hit in hits
    ]


def extractive_answer(query: str, hits: list[dict], memories: list[str]) -> str:
    lines = [f"Question: {query}", "", "Grounded answer:"]
    if memories:
        lines.append("Relevant memory:")
        lines.extend(f"- {memory}" for memory in memories)
    for index, hit in enumerate(hits, start=1):
        text = " ".join(hit["text"].split())
        lines.append(f"- [{index}] {text[:280]} ({hit['filename']}, chunk {hit['chunk_index']})")
    if not hits:
        lines.append("- No indexed evidence was found. Run ingest first.")
    return "\n".join(lines)


def has_llm_config() -> bool:
    return bool(os.getenv("LITELLM_API_KEY") or os.getenv("OPENAI_API_KEY"))


def llm_answer(query: str, hits: list[dict], memories: list[str]) -> str | None:
    try:
        from litellm import completion
    except ImportError:
        return None
    context = "\n\n".join(
        f"[{index}] {hit['filename']} chunk {hit['chunk_index']}: {hit['text']}"
        for index, hit in enumerate(hits, start=1)
    )
    memory_text = "\n".join(f"- {item}" for item in memories)
    try:
        response = completion(
            model=os.getenv("LITELLM_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": "Answer only from provided evidence. Cite source markers like [1].",
                },
                {
                    "role": "user",
                    "content": f"Memory:\n{memory_text}\n\nEvidence:\n{context}\n\nQuestion: {query}",
                },
            ],
            temperature=0.2,
        )
    except Exception:
        return None
    content = response.choices[0].message.content
    return str(content).strip() if content else None
