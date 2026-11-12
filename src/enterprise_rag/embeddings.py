"""Pluggable embedding clients: local stub, OpenAI, and Azure OpenAI."""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import httpx
import numpy as np

from enterprise_rag.config import Settings

_TOKEN = re.compile(r"[A-Za-z0-9_]+")


class EmbeddingClient(Protocol):
    model_name: str
    dimension: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an array of shape (len(texts), dimension), L2-normalised."""


class HashEmbeddingClient:
    """Deterministic hashed bag-of-words embeddings for offline tests and demos."""

    def __init__(self, dimension: int = 128, seed: int = 42) -> None:
        self.model_name = "local-hash-embedding"
        self.dimension = dimension
        self._seed = seed

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float64)
        for i, text in enumerate(texts):
            matrix[i] = self._vectorize(text)
        return matrix

    def _vectorize(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float64)
        tokens = _TOKEN.findall(text.lower())
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(f"{self._seed}:{token}".encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:8], "little") % self.dimension
            sign = 1.0 if digest[8] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector


class OpenAIEmbeddingClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        timeout: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when RAG_PROVIDER=openai")
        self.model_name = model
        self.dimension = -1
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def embed(self, texts: list[str]) -> np.ndarray:
        payload = {"model": self.model_name, "input": texts}
        data = _post_json(
            f"{self._base_url}/embeddings",
            headers={"Authorization": f"Bearer {self._api_key}"},
            payload=payload,
            timeout=self._timeout,
        )
        rows = [item["embedding"] for item in sorted(data["data"], key=lambda row: row["index"])]
        matrix = np.asarray(rows, dtype=np.float64)
        self.dimension = matrix.shape[1]
        return _l2_normalize(matrix)


class AzureOpenAIEmbeddingClient:
    def __init__(
        self,
        *,
        api_key: str,
        endpoint: str,
        deployment: str,
        api_version: str,
        timeout: float = 30.0,
    ) -> None:
        if not api_key or not endpoint:
            raise ValueError(
                "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT are required when RAG_PROVIDER=azure"
            )
        self.model_name = deployment
        self.dimension = -1
        self._api_key = api_key
        self._url = (
            f"{endpoint.rstrip('/')}/openai/deployments/{deployment}/embeddings"
            f"?api-version={api_version}"
        )
        self._timeout = timeout

    def embed(self, texts: list[str]) -> np.ndarray:
        data = _post_json(
            self._url,
            headers={"api-key": self._api_key, "Content-Type": "application/json"},
            payload={"input": texts},
            timeout=self._timeout,
        )
        rows = [item["embedding"] for item in sorted(data["data"], key=lambda row: row["index"])]
        matrix = np.asarray(rows, dtype=np.float64)
        self.dimension = matrix.shape[1]
        return _l2_normalize(matrix)


def build_embedding_client(settings: Settings) -> EmbeddingClient:
    if settings.rag_provider == "openai":
        return OpenAIEmbeddingClient(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
            base_url=settings.openai_base_url,
            timeout=settings.request_timeout_seconds,
        )
    if settings.rag_provider == "azure":
        return AzureOpenAIEmbeddingClient(
            api_key=settings.azure_openai_api_key,
            endpoint=settings.azure_openai_endpoint,
            deployment=settings.azure_openai_embedding_deployment,
            api_version=settings.azure_openai_api_version,
            timeout=settings.request_timeout_seconds,
        )
    return HashEmbeddingClient()


def _post_json(url: str, *, headers: dict, payload: dict, timeout: float) -> dict:
    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms
