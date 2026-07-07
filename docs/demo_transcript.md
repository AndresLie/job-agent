# Demo Transcript

```bash
python -m career_copilot ingest --source data/raw --rebuild
```

Indexes resume notes, project evidence, and a sample job description.

```bash
python -m career_copilot remember --summary "I prefer backend AI engineering roles using Python, retrieval, and evaluation."
```

Stores a durable candidate preference.

```bash
python -m career_copilot ask "Which projects show RAG, memory, and evaluation experience?" --no-llm
```

Returns cited evidence from the RAG project, Pi Memory project, and Hermes
evaluation project.

```bash
python -m career_copilot brief --job data/raw/job_posting.md --write-contract
```

Writes a structured job-fit brief with matched evidence, skill gaps, actions,
citations, and confidence.
