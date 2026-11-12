"""Prompt assembly for grounded answers."""

from __future__ import annotations

import re

from enterprise_rag.vectorstore import ScoredChunk

SYSTEM_PROMPT = (
    "You are an enterprise assistant. Answer the user using only the source excerpts "
    "provided below. If the excerpts are not sufficient, say you do not have enough "
    "information. Cite sources as [n] where n is the excerpt number."
)


def build_messages(query: str, evidence: list[ScoredChunk]) -> list[dict[str, str]]:
    if not evidence:
        context = "No source excerpts were retrieved."
    else:
        blocks = []
        for i, item in enumerate(evidence, start=1):
            source = item.chunk.metadata.get("source", item.chunk.document_id)
            blocks.append(f"[{i}] source={source} chunk={item.chunk.chunk_id}\n{item.chunk.text}")
        context = "\n\n".join(blocks)

    user = (
        f"Source excerpts:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def stub_answer(query: str, evidence: list[ScoredChunk]) -> str:
    """Deterministic extractive answer used by the local stub provider."""
    if not evidence:
        return "I do not have enough information in the knowledge base to answer that."

    query_terms = {token for token in _tokens(query) if len(token) > 2}
    ranked = []
    for item in evidence:
        overlap = len(query_terms.intersection(_tokens(item.chunk.text)))
        ranked.append((overlap, item))
    ranked.sort(key=lambda pair: (pair[0], pair[1].score), reverse=True)

    best_overlap, best = ranked[0]
    source = best.chunk.metadata.get("source", best.chunk.document_id)
    if best_overlap == 0 and best.score < 0.05:
        return "I do not have enough information in the knowledge base to answer that."

    excerpt = _best_sentence(best.chunk.text, query_terms) or best.chunk.text
    if len(excerpt) > 400:
        excerpt = excerpt[:400].rsplit(" ", 1)[0] + "..."
    return f"Based on the retrieved sources: {excerpt} [1] (source: {source})"


def _best_sentence(text: str, query_terms: set[str]) -> str:
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if not parts:
        return text
    scored = []
    for part in parts:
        overlap = len(query_terms.intersection(_tokens(part)))
        scored.append((overlap, -len(part), part))
    scored.sort(reverse=True)
    return scored[0][2] if scored[0][0] else text


def _tokens(text: str) -> set[str]:
    return {part.lower() for part in text.replace("-", " ").split() if part}
