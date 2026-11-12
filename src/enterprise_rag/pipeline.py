"""End-to-end RAG pipeline: ingest, retrieve, and answer."""

from __future__ import annotations

from dataclasses import dataclass, field

from enterprise_rag.chunking import Chunk, chunk_text
from enterprise_rag.config import Settings, get_settings
from enterprise_rag.embeddings import EmbeddingClient, build_embedding_client
from enterprise_rag.llm import ChatClient, build_chat_client
from enterprise_rag.prompts import build_messages
from enterprise_rag.vectorstore import InMemoryVectorStore, ScoredChunk


@dataclass
class Document:
    document_id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class AskResult:
    query: str
    answer: str
    sources: list[dict]
    provider: str


class RAGPipeline:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        embedding_client: EmbeddingClient | None = None,
        chat_client: ChatClient | None = None,
        chunk_size: int = 400,
        chunk_overlap: int = 80,
    ) -> None:
        self.settings = settings or get_settings()
        self.embeddings = embedding_client or build_embedding_client(self.settings)
        self.chat = chat_client or build_chat_client(self.settings)
        self.store = InMemoryVectorStore()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest(self, documents: list[Document]) -> list[Chunk]:
        added: list[Chunk] = []
        for document in documents:
            pieces = chunk_text(
                document.text,
                document_id=document.document_id,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                metadata=document.metadata,
            )
            if not pieces:
                continue
            vectors = self.embeddings.embed([chunk.text for chunk in pieces])
            self.store.add(pieces, vectors)
            added.extend(pieces)
        return added

    def retrieve(self, query: str, top_k: int = 4) -> list[ScoredChunk]:
        query_vector = self.embeddings.embed([query])[0]
        return self.store.search(query_vector, top_k=top_k)

    def ask(self, query: str, top_k: int = 4) -> AskResult:
        evidence = self.retrieve(query, top_k=top_k)
        messages = build_messages(query, evidence)
        answer = self.chat.complete(messages, evidence, query)
        sources = [
            {
                "chunk_id": item.chunk.chunk_id,
                "document_id": item.chunk.document_id,
                "score": round(item.score, 4),
                "source": item.chunk.metadata.get("source", item.chunk.document_id),
                "text": item.chunk.text,
            }
            for item in evidence
        ]
        return AskResult(
            query=query,
            answer=answer,
            sources=sources,
            provider=self.settings.rag_provider,
        )
