import pytest

from enterprise_rag.chunking import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("", document_id="doc") == []
    assert chunk_text("   ", document_id="doc") == []


def test_short_text_is_a_single_chunk():
    chunks = chunk_text("Retain records for seven years.", document_id="policy")
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "policy:0"
    assert "seven years" in chunks[0].text


def test_long_text_is_split_with_overlap():
    sentence = "Privileged access reviews are completed every quarter. "
    text = sentence * 20
    chunks = chunk_text(text, document_id="access", chunk_size=120, chunk_overlap=30)
    assert len(chunks) > 1
    assert chunks[0].index == 0
    assert chunks[-1].index == len(chunks) - 1
    # Overlap means the start of chunk n+1 appears in chunk n
    assert any(token in chunks[0].text for token in chunks[1].text.split()[:4])


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        chunk_text("hello", document_id="x", chunk_size=0)
    with pytest.raises(ValueError):
        chunk_text("hello", document_id="x", chunk_size=10, chunk_overlap=10)
