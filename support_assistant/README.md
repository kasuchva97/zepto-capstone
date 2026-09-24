# Module 3 — Support Assistant

A small, fully offline-testable RAG service: 8 Zepto policy documents, embedded locally
and stored in ChromaDB, orchestrated through a LangGraph `StateGraph` that classifies
each query and either retrieves grounded context or gives a fixed fallback answer,
wrapped in a FastAPI `POST /ask` endpoint with a Pydantic-validated response schema.
Every LLM call is gated behind `MOCK_LLM` (default: deterministic mock, no API key, no
network) — the default, fully offline path.

## Setup

```bash
cd support_assistant
py -3.11 -m venv .venv        # Windows; `python3.11 -m venv .venv` on macOS/Linux
.venv\Scripts\activate        # Windows; `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
```

**Requires Python 3.11** (what this was built and tested against). Newer Pythons
(3.13+) may lack prebuilt wheels for some dependencies here (chromadb,
sentence-transformers, torch) and can hang trying to build them from source — check
`python --version` first, and if it isn't 3.11.x, install Python 3.11 from
[python.org](https://www.python.org/downloads/) alongside your existing version
rather than swapping `py -3.11` for plain `python` above.

The `all-MiniLM-L6-v2` embedding model
downloads once on first use (needs internet) and is cached locally afterward — this
is the same one-time-download-then-cache pattern as `sns.load_dataset` in Module 2.

## Run locally

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```
Open `http://127.0.0.1:8000/` for a small demo page, or call the API directly:
```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"query\": \"How much does standard delivery cost?\"}"
```

## Run tests

```bash
python -m pytest -q
```
39 tests: the `classify_intent` keyword heuristic (every required keyword + a negative
case), retrieval correctness against the real corpus (right document ranks top for a
topically matching query), the full LangGraph flow for both routing branches, the
`/ask` endpoint via FastAPI's `TestClient`, and the optional real-LLM path's
retry-on-invalid-output logic (exercised via a monkeypatched fake LLM call, so it's
verified without needing a Groq API key).

## Example `/ask` calls (recorded with `MOCK_LLM` at its default)

**A query that triggers retrieval** (`policy_question` — contains "delivery"):
```json
// POST /ask {"query": "How much does standard delivery cost?"}
{
  "answer": "Based on the retrieved context: Delivery Policy: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order vol",
  "sources": ["doc_01", "doc_05", "doc_02"],
  "confidence": 1.0
}
```

**A query that does not trigger retrieval** (`general_question` — no policy keyword):
```json
// POST /ask {"query": "What is the capital of France?"}
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

## `MOCK_LLM` toggle

Left unset, or `MOCK_LLM=1` (the default): the standard offline path. No LLM API key
or network call anywhere — `classify_intent` uses a keyword heuristic, and both
answer nodes return deterministic canned text. Retrieval (embedding the query +
querying ChromaDB) is **not** mocked in either mode — it's always real, since it
needs no API key.

`MOCK_LLM=0` (optional extension): requires `GROQ_API_KEY` in the
environment (Groq's free tier, console.groq.com). `classify_intent` asks the LLM to
classify instead of the keyword heuristic; `retrieve_and_answer` prompts the LLM
(using the structured template in `src/prompts.py`) to answer grounded in the
retrieved chunks, requesting JSON matching the response schema and retrying up to 2
additional times with a corrective instruction if the output fails to parse/validate
(`src/llm.py:generate_structured_policy_answer_llm`) before giving up with a clearly
marked error response; `direct_answer` prompts the LLM directly with no retrieval.
This path was **not exercised against a real Groq account** here (no API key was
configured) — its retry logic is verified in `tests/test_llm_retry.py` by
monkeypatching the low-level `_call_groq` call, and the default `MOCK_LLM=1` path
works fully independently of it.

## RAG pipeline architecture

**Ingestion.** `src/ingest.py:load_documents()` reads the 8 files under `docs/`,
one chunk per document (each is already a single short paragraph, so no further
splitting is needed). Each chunk keeps its document id (`doc_01`...`doc_08`) and
source filename as metadata.

**Embedding.** `src/ingest.py:get_embedder()` loads `sentence-transformers`'
`all-MiniLM-L6-v2` locally (cached after the first download; `HF_HUB_OFFLINE` is tried
first so subsequent runs skip the network cache-check round-trip entirely).
`ingest_documents()` embeds all 8 chunks and writes them into a ChromaDB
`PersistentClient` collection named `zepto_policies` (`hnsw:space="cosine"`), stored
under `chroma_store/` (rebuilt from `docs/` on every app startup, so it's never
committed — see `.gitignore`).

**Retrieval.** `src/ingest.py:retrieve_top_k()` embeds the incoming query with the
same model and queries the `zepto_policies` collection for its top-3 nearest chunks by
cosine similarity. This is called from the `retrieve_and_answer` node in
`src/graph.py` and runs for real in **both** `MOCK_LLM` modes, since it needs no API
key.

**Generation.** `src/graph.py` is a LangGraph `StateGraph` (`SupportState`
`TypedDict`: `query, intent, answer, sources, confidence`) with 3 nodes:
- `classify_intent` — keyword heuristic (mock) or LLM call (`MOCK_LLM=0`) → sets `intent`.
- A conditional edge (`_route_on_intent`, independent of `MOCK_LLM`) routes
  `policy_question` → `retrieve_and_answer`, `general_question` → `direct_answer`.
- `retrieve_and_answer` — always retrieves for real (see above); then either returns
  the canned `f"Based on the retrieved context: {snippet}"` template (mock) or prompts
  the LLM with `src/prompts.py:build_policy_answer_prompt()` (`MOCK_LLM=0`).
- `direct_answer` — returns a fixed canned string (mock) or prompts the LLM directly
  (`MOCK_LLM=0`), with no retrieval either way.

`main.py` builds the graph once at FastAPI startup (`lifespan`, after
`ingest_documents()`), and `POST /ask` runs `src/graph.py:run_support_graph()` per
request, validating the final state against `src/schemas.py:AskResponse` (`answer:
str, sources: list[str], confidence: float 0-1`) before returning it.

**Data flow, end to end:** `docs/*.txt` → `load_documents` → `get_embedder().encode`
→ ChromaDB `zepto_policies` collection → (per request) `POST /ask` → `classify_intent`
→ (conditional edge) → `retrieve_and_answer` (re-embeds the query, queries ChromaDB,
generates the answer) *or* `direct_answer` (generates the answer, no retrieval) →
`AskResponse` → JSON back to the client.

## Docker

```bash
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
# then: curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" -d "{\"query\": \"How do I cancel an order?\"}"
```

**Verified.** Built and run with Docker Desktop 4.91 (WSL 2 backend) on Windows 10:
the image builds cleanly, the container starts in about 12 seconds, and `POST /ask`
returns the expected response for both a policy question (retrieval, with sources) and
a general question (fixed fallback), and `GET /` serves the demo page.

Notes on the image:
- It installs the **CPU-only** build of PyTorch first. A plain `pip install
  sentence-transformers` on Linux pulls the CUDA build plus several GB of NVIDIA
  libraries this app never uses (it only embeds a few short strings on CPU).
- The embedding model is downloaded at build time, so the running container needs no
  network access in the default `MOCK_LLM` mode.
- `build-essential` is installed as a fallback in case `chromadb`'s `hnswlib`
  dependency has to compile from source on the slim base image.

The optional Hugging Face Spaces deployment was not attempted.
