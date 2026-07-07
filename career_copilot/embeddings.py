from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable, Protocol


class Embedder(Protocol):
    name: str

    def embed(self, text: str) -> list[float]:
        ...


class HashingEmbedder:
    name = "hashing-v1"

    def __init__(self, vector_size: int = 512) -> None:
        self.vector_size = vector_size

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.vector_size
        for token in tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.vector_size
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign * (1.0 + math.log1p(len(token)))
        return normalize(vector)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.name = f"sentence-transformers:{model_name}"
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        row = self._model.encode([text], normalize_embeddings=True)[0]
        return [float(value) for value in row]


def build_embedder(name: str = "hashing") -> Embedder:
    if name == "sentence-transformers":
        return SentenceTransformerEmbedder()
    return HashingEmbedder()


def tokenize(text: str) -> Iterable[str]:
    normalized = text.casefold()
    words = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", normalized)
    joined = re.sub(r"\s+", "", normalized)
    grams = [joined[index : index + 3] for index in range(max(len(joined) - 2, 0))]
    return words + grams


def normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return values
    return [value / norm for value in values]


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    return sum(a * b for a, b in zip(left, right))
