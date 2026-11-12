from enterprise_rag.pipeline import Document, RAGPipeline


def test_ask_grounds_answer_in_retrieved_policy(loaded_pipeline: RAGPipeline):
    result = loaded_pipeline.ask("How long must customer transaction records be retained?", top_k=3)
    assert result.provider == "stub"
    assert result.sources
    assert "seven" in result.answer.lower() or "7" in result.answer
    assert any(source["document_id"] == "records" for source in result.sources)


def test_ask_without_documents_admits_insufficient_information():
    result = RAGPipeline().ask("What is the retention period?")
    assert "not have enough information" in result.answer.lower()
    assert result.sources == []


def test_ingest_skips_empty_documents():
    pipeline = RAGPipeline()
    chunks = pipeline.ingest([Document(document_id="empty", text="   ", metadata={})])
    assert chunks == []
    assert len(pipeline.store) == 0
