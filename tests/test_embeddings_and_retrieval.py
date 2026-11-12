import numpy as np

from enterprise_rag.chunking import Chunk
from enterprise_rag.embeddings import HashEmbeddingClient
from enterprise_rag.vectorstore import InMemoryVectorStore


def test_hash_embeddings_are_deterministic_and_normalized():
    client = HashEmbeddingClient(dimension=64)
    first = client.embed(["data retention period"])
    second = client.embed(["data retention period"])
    assert first.shape == (1, 64)
    np.testing.assert_allclose(first, second)
    assert np.isclose(np.linalg.norm(first[0]), 1.0)


def test_similar_texts_score_higher_than_unrelated_texts():
    client = HashEmbeddingClient(dimension=128)
    query, close, other = client.embed(
        [
            "Where is EU personal data stored?",
            "EU personal data is stored in the West Europe Azure region.",
            "Temporary elevated access expires after 24 hours.",
        ]
    )
    close_score = float(query @ close)
    other_score = float(query @ other)
    assert close_score > other_score


def test_vector_store_returns_top_k_by_cosine_similarity():
    client = HashEmbeddingClient(dimension=128)
    chunks = [
        Chunk("a:0", "a", "EU personal data is stored in West Europe.", 0, {"source": "a"}),
        Chunk("b:0", "b", "Security logs are retained for 18 months.", 0, {"source": "b"}),
        Chunk("c:0", "c", "Severity-1 incidents must be acknowledged within 15 minutes.", 0, {"source": "c"}),
    ]
    store = InMemoryVectorStore()
    store.add(chunks, client.embed([chunk.text for chunk in chunks]))
    query = client.embed(["Where is EU personal data stored?"])[0]
    results = store.search(query, top_k=2)
    assert len(results) == 2
    assert results[0].chunk.document_id == "a"
    assert results[0].score >= results[1].score
