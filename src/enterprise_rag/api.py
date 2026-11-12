"""FastAPI app exposing ingest, ask, and health endpoints."""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from enterprise_rag.pipeline import Document, RAGPipeline

app = FastAPI(
    title="enterprise-rag-azure",
    description="Reference RAG API with pluggable Azure OpenAI / OpenAI / local stub clients.",
    version="0.1.0",
)


class IngestDocument(BaseModel):
    id: str = Field(..., min_length=1)
    text: str
    metadata: dict = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[IngestDocument]


class IngestResponse(BaseModel):
    ingested_documents: int
    ingested_chunks: int
    store_size: int


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=4, ge=1, le=20)


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: list[dict]
    provider: str


@lru_cache
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


@app.get("/health")
def health() -> dict:
    pipeline = get_pipeline()
    return {
        "status": "ok",
        "provider": pipeline.settings.rag_provider,
        "store_size": len(pipeline.store),
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    if not request.documents:
        raise HTTPException(status_code=400, detail="documents must not be empty")
    pipeline = get_pipeline()
    chunks = pipeline.ingest(
        [
            Document(
                document_id=item.id,
                text=item.text,
                metadata={**item.metadata, "source": item.metadata.get("source", item.id)},
            )
            for item in request.documents
        ]
    )
    return IngestResponse(
        ingested_documents=len(request.documents),
        ingested_chunks=len(chunks),
        store_size=len(pipeline.store),
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    result = get_pipeline().ask(request.query, top_k=request.top_k)
    return AskResponse(
        query=result.query,
        answer=result.answer,
        sources=result.sources,
        provider=result.provider,
    )
