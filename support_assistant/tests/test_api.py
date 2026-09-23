import os

os.environ.setdefault("MOCK_LLM", "1")

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Points the app's lifespan at a throwaway ChromaDB directory (via
    CHROMA_PERSIST_DIR) instead of the default chroma_store/. Without this,
    running this test suite while a real dev server is up would delete and
    rebuild that server's collection out from under it -- ingest_documents()
    always deletes + recreates the collection, and the running server's
    in-memory reference to the old one would then 500 on every request.
    (This is exactly what happened once during development -- see git log.)
    """
    persist_dir = tmp_path_factory.mktemp("test_api_chroma_store")
    previous = os.environ.get("CHROMA_PERSIST_DIR")
    os.environ["CHROMA_PERSIST_DIR"] = str(persist_dir)
    try:
        with TestClient(app) as c:  # runs the lifespan (ingest_documents + build graph) once
            yield c
    finally:
        if previous is None:
            os.environ.pop("CHROMA_PERSIST_DIR", None)
        else:
            os.environ["CHROMA_PERSIST_DIR"] = previous


class TestAskEndpoint:
    def test_policy_question_returns_valid_schema(self, client):
        resp = client.post("/ask", json={"query": "How much does standard delivery cost?"})
        assert resp.status_code == 200
        data = resp.json()
        assert set(data.keys()) == {"answer", "sources", "confidence"}
        assert isinstance(data["answer"], str) and data["answer"]
        assert isinstance(data["sources"], list) and len(data["sources"]) > 0
        assert 0.0 <= data["confidence"] <= 1.0

    def test_general_question_returns_empty_sources(self, client):
        resp = client.post("/ask", json={"query": "What's the weather like today?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["sources"] == []
        assert data["answer"] == "I can only answer questions about Zepto policies right now."

    def test_missing_query_field_returns_422(self, client):
        resp = client.post("/ask", json={})
        assert resp.status_code == 422

    def test_demo_page_is_served(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Zepto Support Assistant" in resp.text
