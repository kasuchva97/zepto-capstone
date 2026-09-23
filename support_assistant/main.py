"""FastAPI wrapper around the LangGraph support-assistant graph.

POST /ask -- the graded endpoint (AskRequest -> AskResponse).
GET  /    -- a small light-theme demo page for trying /ask by hand.
"""
from __future__ import annotations

import os
import pathlib
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.graph import build_support_graph, run_support_graph
from src.ingest import DEFAULT_PERSIST_DIR, ingest_documents
from src.schemas import AskRequest, AskResponse

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # CHROMA_PERSIST_DIR lets a test suite (or a second instance) point at an
    # isolated directory instead of colliding with a real running server's
    # on-disk collection -- ingest_documents() deletes and recreates the
    # collection on every startup, which corrupts another process's
    # in-memory reference to it if they share the same directory.
    persist_dir = pathlib.Path(os.environ.get("CHROMA_PERSIST_DIR", str(DEFAULT_PERSIST_DIR)))
    collection = ingest_documents(persist_dir=persist_dir)  # rebuilds the ChromaDB collection from docs/ on startup
    _state["graph"] = build_support_graph(collection)
    yield
    _state.clear()


app = FastAPI(title="Zepto Support Assistant", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def demo_page():
    return FileResponse("static/index.html")


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    result = run_support_graph(_state["graph"], request.query)
    return AskResponse(answer=result["answer"], sources=result["sources"], confidence=result["confidence"])
