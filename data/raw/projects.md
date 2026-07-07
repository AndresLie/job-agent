# Project Evidence

## Personal RAG for Social Media Trend Analysis

Built a local RAG system over public marketing reports and articles. The
pipeline cleaned PDF and Markdown sources, chunked text with paragraph metadata,
generated embeddings, stored chunks in a vector index, and answered questions
with citations.

## Pi Memory

Implemented persistent memory for a coding agent. The system captured durable
observations, stored them in JSON, retrieved relevant memories with BM25, and
injected them into later sessions under a token budget. A hybrid BM25 plus
embedding path improved lexical-gap benchmark cases.

## Hermes Skill Evaluation

Built verifiable LLM skills for Text2SQL, code authoring, bug hunting, and test
generation. The skills wrote result files through deterministic scripts, used
schema validation, and reported objective outcomes through local test runners.
