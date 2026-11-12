"""In-memory vector store with cosine similarity search."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from enterprise_rag.chunking import Chunk


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not len(chunks):
            return
        self._chunks.extend(chunks)
        if self._vectors is None:
            self._vectors = np.asarray(vectors, dtype=np.float64)
        else:
            self._vectors = np.vstack([self._vectors, np.asarray(vectors, dtype=np.float64)])

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> list[ScoredChunk]:
        if self._vectors is None or not self._chunks:
            return []
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        vector = np.asarray(query_vector, dtype=np.float64).reshape(-1)
        scores = self._vectors @ vector
        k = min(top_k, len(self._chunks))
        # argpartition is enough for top-k; sort the shortlist descending
        top_indices = np.argpartition(scores, -k)[-k:]
        ranked = sorted(top_indices, key=lambda idx: float(scores[idx]), reverse=True)
        return [ScoredChunk(chunk=self._chunks[idx], score=float(scores[idx])) for idx in ranked]
