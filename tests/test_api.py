def test_health_ok(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["provider"] == "stub"


def test_ingest_and_ask_roundtrip(api_client):
    ingest = api_client.post(
        "/ingest",
        json={
            "documents": [
                {
                    "id": "policy",
                    "text": "Customer transaction records must be retained for seven years.",
                    "metadata": {"source": "policy.md"},
                }
            ]
        },
    )
    assert ingest.status_code == 200
    assert ingest.json()["ingested_chunks"] >= 1

    asked = api_client.post("/ask", json={"query": "What is the retention period for transaction records?", "top_k": 3})
    assert asked.status_code == 200
    body = asked.json()
    assert "seven" in body["answer"].lower() or "retain" in body["answer"].lower()
    assert body["sources"]
    assert body["provider"] == "stub"


def test_ingest_rejects_empty_document_list(api_client):
    response = api_client.post("/ingest", json={"documents": []})
    assert response.status_code == 400
