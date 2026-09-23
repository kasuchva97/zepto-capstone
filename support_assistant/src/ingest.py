"""Loads the 8 Zepto policy documents, chunks them (one chunk per document --
each is already a single short paragraph, so no further splitting is
needed), embeds each chunk locally with sentence-transformers'
all-MiniLM-L6-v2, and stores the embeddings in a ChromaDB collection using
cosine similarity.
"""
from __future__ import annotations

import functools
import os
import pathlib

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

DOCS_DIR = pathlib.Path(__file__).resolve().parents[1] / "docs"
DEFAULT_PERSIST_DIR = pathlib.Path(__file__).resolve().parents[1] / "chroma_store"
COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_SETTINGS = Settings(anonymized_telemetry=False)


@functools.lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Tries HF_HUB_OFFLINE first -- if the model is already cached locally
    (the common case after the first run), this skips the network
    cache-validation round-trip entirely, which otherwise adds a slow
    multi-retry delay whenever DNS/network is flaky. Falls back to a normal
    (online) load on the very first run, when nothing is cached yet.
    """
    previous = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        return SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception:
        if previous is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous
        return SentenceTransformer(EMBEDDING_MODEL_NAME)


def load_documents(docs_dir: pathlib.Path = DOCS_DIR) -> list[dict]:
    """One chunk per document file -- returns [{id, text, source}, ...]."""
    chunks = []
    for path in sorted(docs_dir.glob("doc_*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        chunks.append({"id": path.stem, "text": text, "source": path.name})
    return chunks


def get_chroma_collection(persist_dir: pathlib.Path = DEFAULT_PERSIST_DIR):
    client = chromadb.PersistentClient(path=str(persist_dir), settings=CHROMA_SETTINGS)
    return client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def ingest_documents(
    docs_dir: pathlib.Path = DOCS_DIR, persist_dir: pathlib.Path = DEFAULT_PERSIST_DIR
):
    """(Re)builds the ChromaDB collection from scratch from the doc files."""
    client = chromadb.PersistentClient(path=str(persist_dir), settings=CHROMA_SETTINGS)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    chunks = load_documents(docs_dir)
    embedder = get_embedder()
    embeddings = embedder.encode([c["text"] for c in chunks]).tolist()

    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embeddings,
        metadatas=[{"source": c["source"]} for c in chunks],
    )
    return collection


def retrieve_top_k(collection, query: str, k: int = 3) -> list[dict]:
    """Embeds `query` and retrieves the top-k most similar chunks (cosine similarity)."""
    embedder = get_embedder()
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=k)

    hits = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]  # cosine distance (1 - cosine similarity)
        hits.append(
            {
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "distance": distance,
                "similarity": 1 - distance,
            }
        )
    return hits
