"""CLI: ingest a document (or the sample file) and answer a question offline."""

from __future__ import annotations

import argparse
from pathlib import Path

from enterprise_rag.pipeline import Document, RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a one-shot RAG query.")
    parser.add_argument("query", help="Question to ask against the ingested document")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "sample_policy.md",
        help="Path to a text or markdown document to ingest",
    )
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    if not args.data.is_file():
        raise SystemExit(f"Document not found: {args.data}")

    pipeline = RAGPipeline()
    text = args.data.read_text(encoding="utf-8")
    pipeline.ingest(
        [Document(document_id=args.data.stem, text=text, metadata={"source": args.data.name})]
    )
    result = pipeline.ask(args.query, top_k=args.top_k)
    print(result.answer)
    print()
    print("Sources:")
    for source in result.sources:
        score = source["score"]
        print(f"- {source['chunk_id']} (score={score:.3f}) {source['text'][:160]}")


if __name__ == "__main__":
    main()
