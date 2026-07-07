from __future__ import annotations

import json
import os

import requests

from .memory import MemoryStore
from .vector_store import JsonVectorStore
from .embeddings import Embedder


NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_DEFAULT_MODEL = "google/gemma-4-31b-it"


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
            "category": hit.get("category", "general"),
            "source_path": hit.get("source_path", hit["filename"]),
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
        category = hit.get("category", "general")
        lines.append(f"- [{index}] {text[:280]} ({category}/{hit['filename']}, chunk {hit['chunk_index']})")
    if not hits:
        lines.append("- No indexed evidence was found. Run ingest first.")
    return "\n".join(lines)


def has_llm_config() -> bool:
    return bool(os.getenv("NVIDIA_API_KEY"))


def llm_answer(query: str, hits: list[dict], memories: list[str]) -> str | None:
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        return None

    context = "\n\n".join(
        f"[{index}] {hit.get('category', 'general')}/{hit['filename']} "
        f"chunk {hit['chunk_index']}: {hit['text']}"
        for index, hit in enumerate(hits, start=1)
    )
    memory_text = "\n".join(f"- {item}" for item in memories)
    payload = {
        "model": os.getenv("NVIDIA_MODEL", NVIDIA_DEFAULT_MODEL),
        "messages": [
            {
                "role": "system",
                "content": "Answer only from provided evidence. Cite source markers like [1].",
            },
            {
                "role": "user",
                "content": f"Memory:\n{memory_text}\n\nEvidence:\n{context}\n\nQuestion: {query}",
            },
        ],
        "max_tokens": int(os.getenv("NVIDIA_MAX_TOKENS", "2048")),
        "temperature": float(os.getenv("NVIDIA_TEMPERATURE", "1.0")),
        "top_p": float(os.getenv("NVIDIA_TOP_P", "0.95")),
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    invoke_url = os.getenv("NVIDIA_BASE_URL", NVIDIA_BASE_URL).rstrip("/") + "/chat/completions"
    try:
        response = requests.post(
            invoke_url,
            headers=headers,
            json=payload,
            timeout=float(os.getenv("NVIDIA_TIMEOUT", "120")),
        )
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, json.JSONDecodeError, ValueError):
        return None

    choices = data.get("choices") or []
    if not choices:
        return None
    content = choices[0].get("message", {}).get("content")
    return str(content).strip() if content else None
