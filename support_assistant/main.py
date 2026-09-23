"""FastAPI wrapper around the LangGraph support-assistant graph.

POST /ask -- the graded endpoint (AskRequest -> AskResponse).
GET  /    -- a small light-theme demo page for trying /ask by hand.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.graph import build_support_graph, run_support_graph
from src.ingest import ingest_documents
from src.schemas import AskRequest, AskResponse

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    collection = ingest_documents()  # rebuilds the ChromaDB collection from docs/ on startup
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
