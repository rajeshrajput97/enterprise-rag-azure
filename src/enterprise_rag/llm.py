"""Pluggable chat clients: local stub, OpenAI, and Azure OpenAI."""

from __future__ import annotations

from typing import Protocol

import httpx

from enterprise_rag.config import Settings
from enterprise_rag.prompts import stub_answer
from enterprise_rag.vectorstore import ScoredChunk


class ChatClient(Protocol):
    model_name: str

    def complete(self, messages: list[dict[str, str]], evidence: list[ScoredChunk], query: str) -> str:
        """Return a model completion. `evidence` and `query` are used by the stub only."""


class StubChatClient:
    model_name = "local-stub-chat"

    def complete(self, messages: list[dict[str, str]], evidence: list[ScoredChunk], query: str) -> str:
        # messages are accepted for interface parity with hosted clients
        _ = messages
        return stub_answer(query, evidence)


class OpenAIChatClient:
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
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def complete(self, messages: list[dict[str, str]], evidence: list[ScoredChunk], query: str) -> str:
        _ = evidence, query
        data = _post_json(
            f"{self._base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            payload={"model": self.model_name, "messages": messages, "temperature": 0},
            timeout=self._timeout,
        )
        return data["choices"][0]["message"]["content"]


class AzureOpenAIChatClient:
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
        self._url = (
            f"{endpoint.rstrip('/')}/openai/deployments/{deployment}/chat/completions"
            f"?api-version={api_version}"
        )
        self._api_key = api_key
        self._timeout = timeout

    def complete(self, messages: list[dict[str, str]], evidence: list[ScoredChunk], query: str) -> str:
        _ = evidence, query
        data = _post_json(
            self._url,
            headers={"api-key": self._api_key, "Content-Type": "application/json"},
            payload={"messages": messages, "temperature": 0},
            timeout=self._timeout,
        )
        return data["choices"][0]["message"]["content"]


def build_chat_client(settings: Settings) -> ChatClient:
    if settings.rag_provider == "openai":
        return OpenAIChatClient(
            api_key=settings.openai_api_key,
            model=settings.openai_chat_model,
            base_url=settings.openai_base_url,
            timeout=settings.request_timeout_seconds,
        )
    if settings.rag_provider == "azure":
        return AzureOpenAIChatClient(
            api_key=settings.azure_openai_api_key,
            endpoint=settings.azure_openai_endpoint,
            deployment=settings.azure_openai_chat_deployment,
            api_version=settings.azure_openai_api_version,
            timeout=settings.request_timeout_seconds,
        )
    return StubChatClient()


def _post_json(url: str, *, headers: dict, payload: dict, timeout: float) -> dict:
    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
