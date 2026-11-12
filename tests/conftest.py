import os

# Force the offline stub before settings are imported by the app module.
os.environ["RAG_PROVIDER"] = "stub"

import pytest

from enterprise_rag.api import get_pipeline
from enterprise_rag.pipeline import Document, RAGPipeline


@pytest.fixture
def pipeline() -> RAGPipeline:
    return RAGPipeline()


@pytest.fixture
def loaded_pipeline(pipeline: RAGPipeline) -> RAGPipeline:
    pipeline.ingest(
        [
            Document(
                document_id="records",
                text=(
                    "Customer transaction records must be retained for seven (7) years "
                    "from the date of the transaction. Security logs are retained for 18 months."
                ),
                metadata={"source": "policy.md"},
            ),
            Document(
                document_id="residency",
                text=(
                    "Personal data of customers in the European Union is stored in the "
                    "West Europe Azure region. Personal data of customers in India is stored "
                    "in the Central India Azure region."
                ),
                metadata={"source": "residency.md"},
            ),
        ]
    )
    return pipeline


@pytest.fixture
def api_client():
    get_pipeline.cache_clear()
    from fastapi.testclient import TestClient
    from enterprise_rag.api import app

    with TestClient(app) as client:
        yield client
    get_pipeline.cache_clear()
