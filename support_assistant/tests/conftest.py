import os

os.environ.setdefault("MOCK_LLM", "1")  # tests always exercise the default offline mock path

import pytest

from src.graph import build_support_graph
from src.ingest import ingest_documents


@pytest.fixture(scope="session")
def collection(tmp_path_factory):
    """A real ChromaDB collection, embedded from the real docs/, but written
    to a throwaway temp directory so tests never touch the app's own
    chroma_store/."""
    persist_dir = tmp_path_factory.mktemp("chroma_test_store")
    return ingest_documents(persist_dir=persist_dir)


@pytest.fixture(scope="session")
def compiled_graph(collection):
    return build_support_graph(collection)
