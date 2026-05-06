"""openRAG middleware server — drop-in RAG quality gate API.

Start:
    python server.py --model /path/to/model.gguf

Endpoints:
    POST /check         — check if context is relevant for a question
    POST /retrieve      — retrieve from knowledge base with quality gate
    POST /query         — full pipeline: retrieve + gate + generate
    GET  /health        — server health check
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from openrag import EntropyGate, OpenRAGPipeline

app = FastAPI(title="openRAG", version="0.1.0")

gate: EntropyGate | None = None
pipeline: OpenRAGPipeline | None = None


class CheckRequest(BaseModel):
    question: str
    context: str
    control_context: str | None = None


class CheckResponse(BaseModel):
    passed: bool
    confidence: str
    delta: float
    h_bare: float
    h_with_context: float
    h_control: float | None


class RetrieveRequest(BaseModel):
    question: str
    top_k: int = 3


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": gate is not None}


@app.post("/check", response_model=CheckResponse)
def check(req: CheckRequest):
    """Check if a context passage is relevant for answering a question."""
    result = gate.check(req.question, req.context, req.control_context)
    return CheckResponse(
        passed=result.passed,
        confidence=result.confidence,
        delta=result.delta,
        h_bare=result.h_bare,
        h_with_context=result.h_with_context,
        h_control=result.h_control,
    )


@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    """Retrieve from knowledge base with entropy quality gating."""
    results = pipeline.query(req.question, top_k=req.top_k)
    return results.to_dict()


@app.post("/query")
def query(req: QueryRequest):
    """Full pipeline: check retrieval need, retrieve, gate, generate."""
    result = pipeline.query(req.question, top_k=req.top_k)
    return result.to_dict()


def main():
    parser = argparse.ArgumentParser(description="openRAG middleware server")
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--docs", nargs="*", help="Text files to load into knowledge base")
    parser.add_argument("--threshold", type=float, default=0.2, help="Gate threshold (nat)")
    args = parser.parse_args()

    global gate, pipeline
    pipeline = OpenRAGPipeline(args.model, gate_threshold=args.threshold)
    gate = pipeline.gate

    if args.docs:
        pipeline.add_files(args.docs)
        print(f"Loaded {len(args.docs)} documents into knowledge base")

    print(f"Starting openRAG server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
