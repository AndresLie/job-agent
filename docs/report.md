# AI Job Copilot Report

## Design Decisions

The project is intentionally local-first. The hashing embedder makes the
default demo reproducible without credentials, while the optional
sentence-transformer path provides a higher-quality retrieval upgrade.

The system keeps memory separate from document evidence. Memory stores durable
preferences and facts; the vector index stores cited project evidence. This
separation makes answers auditable and avoids treating private preferences as
source documents.

The job brief command writes a validated JSON contract. That design comes from
agent evaluation practice: downstream graders and applications should read
structured artifacts instead of relying on conversational prose.

## Evaluation

`python -m career_copilot evaluate` reports Recall@k, MRR, and nDCG@k over the
sample benchmark. The benchmark is small by design, but it demonstrates the
measurement loop needed for larger portfolio datasets.

## Failure Modes

- Hashing retrieval can miss semantic matches when the wording is very
  different.
- The sample corpus is small; a real candidate should add more project notes,
  resume bullets, and job descriptions.
- Without an LLM key, answers are extractive rather than polished. This is a
  deliberate reproducibility tradeoff.

## Future Improvements

- Add ChromaDB persistence as a selectable backend.
- Add reranking for job-evidence matching.
- Add a Streamlit demo once the CLI behavior is stable.
- Track benchmark changes over time in a metrics file.
