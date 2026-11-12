"""Recursive character chunking with overlap."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    index: int
    metadata: dict = field(default_factory=dict)


def chunk_text(
    text: str,
    *,
    document_id: str,
    chunk_size: int = 400,
    chunk_overlap: int = 80,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Split text into overlapping windows.

    Prefers paragraph and sentence boundaries when they fall inside the window.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    cleaned = _normalize(text)
    if not cleaned:
        return []

    meta = dict(metadata or {})
    chunks: list[Chunk] = []
    start = 0
    index = 0
    length = len(cleaned)

    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            end = _prefer_break(cleaned, start, end)
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}:{index}",
                    document_id=document_id,
                    text=piece,
                    index=index,
                    metadata=meta,
                )
            )
            index += 1
        if end >= length:
            break
        start = max(end - chunk_overlap, start + 1)

    return chunks


def _normalize(text: str) -> str:
    if not text:
        return ""
    unified = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(line.split()) for line in unified.split("\n")]
    # Keep paragraph structure; drop leading/trailing empty lines
    return "\n".join(lines).strip()


def _prefer_break(text: str, start: int, end: int) -> int:
    window = text[start:end]
    for separator in ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "):
        pos = window.rfind(separator)
        # Avoid tiny tails that would create almost-empty next windows
        if pos >= max(40, len(window) // 4):
            return start + pos + len(separator)
    return end
