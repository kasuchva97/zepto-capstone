FROM python:3.11-slim

WORKDIR /app

# build-essential is a safety net: chromadb's hnswlib dependency normally
# installs from a prebuilt wheel, but if no wheel matches this exact base
# image, pip falls back to compiling it from source, which needs a C
# compiler that python:3.11-slim doesn't include by default.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install the CPU-only build of PyTorch first. On Linux, a plain
# `pip install sentence-transformers` pulls the default CUDA build of torch
# plus several GB of NVIDIA libraries this app never uses (it only embeds a
# few short strings on CPU). Installing torch from the CPU index up front
# keeps the image small and the build fast; the requirements step below then
# sees torch as already satisfied.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY docs/ ./docs/
COPY src/ ./src/
COPY static/ ./static/
COPY main.py .

# Pre-download the embedding model at build time so container startup
# doesn't need network access (and the default MOCK_LLM=1 path never needs
# an LLM API key either -- only the optional MOCK_LLM=0 extension does).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
