# AI Job Copilot

AI Job Copilot is a local, portfolio-ready agentic RAG system for turning
resume notes, project evidence, and job descriptions into cited job-fit briefs.

It combines three AI engineering patterns:

- RAG over local evidence with citations.
- Durable memory for user preferences and career facts.
- Deterministic JSON contracts and retrieval evaluation.

The default path uses a local hashing embedder, so the demo runs without API
keys or model downloads. Optional LiteLLM and sentence-transformer paths are
available when configured.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python -m career_copilot ingest --source data/raw --rebuild
python -m career_copilot remember --summary "I prefer backend AI engineering roles using Python, retrieval, and evaluation." --tags career,preference
python -m career_copilot ask "Which projects show RAG, memory, and evaluation experience?" --no-llm
python -m career_copilot brief --job data/raw/job_posting.md --write-contract
python -m career_copilot evaluate
```

`--rebuild` resets the demo storage, including vector records and memory.

## What It Demonstrates

- Document ingestion, cleaning, chunking, and citation tracking.
- Retrieval scoring with semantic hashing plus lexical reranking.
- Persistent memory with deterministic BM25 retrieval.
- Structured job-fit output with a validated JSON result contract.
- Objective retrieval metrics: Recall@k, MRR, and nDCG@k.

## CLI

```bash
python -m career_copilot ingest --source data/raw --rebuild
python -m career_copilot ask "How should I pitch my AI engineering projects?" --no-llm
python -m career_copilot remember --summary "Target roles should emphasize evaluation and reliable AI systems."
python -m career_copilot brief --job data/raw/job_posting.md
python -m career_copilot evaluate --k 5
```

## Result Contract

When `AI_JOB_COPILOT_RESULT_PATH` is set, `brief --write-contract` writes a JSON
object with:

- `job_title`
- `fit_score`
- `matched_evidence`
- `skill_gaps`
- `recommended_actions`
- `citations`
- `confidence`

This mirrors production agent workflows where conversational text is not the
source of truth; a validated artifact is.
