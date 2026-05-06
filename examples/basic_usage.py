"""openRAG basic usage example.

Prerequisites:
    1. Download a GGUF model (tested with Qwen2.5-3B Q4_K_M)
    2. pip install llama-cpp-python numpy

Usage:
    python basic_usage.py --model /path/to/model.gguf --docs doc1.txt doc2.txt
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openrag import EntropyGate, TFIDFRetriever, OpenRAGPipeline


def demo_gate(model_path: str):
    """Demo: quality gate on a single question/context pair."""
    print("=" * 60)
    print("DEMO 1: EntropyGate — single context check")
    print("=" * 60)

    gate = EntropyGate(model_path)

    question = "What is the capital of Mongolia?"
    relevant = "Ulaanbaatar is the capital and largest city of Mongolia. It has a population of approximately 1.5 million people."
    irrelevant = "The Pacific Ocean is the largest and deepest ocean on Earth, covering more than 30% of the planet's surface."

    print(f"\nQuestion: {question}")
    print(f"\n--- Relevant context ---")
    r1 = gate.check(question, relevant, irrelevant)
    print(f"  Passed: {r1.passed}  Confidence: {r1.confidence}")
    print(f"  H_bare: {r1.h_bare:.4f}  H_with_ctx: {r1.h_with_context:.4f}  Delta: {r1.delta:+.4f}")

    print(f"\n--- Irrelevant context (control) ---")
    r2 = gate.check(question, irrelevant, relevant)
    print(f"  Passed: {r2.passed}  Confidence: {r2.confidence}")
    print(f"  H_bare: {r2.h_bare:.4f}  H_with_ctx: {r2.h_with_context:.4f}  Delta: {r2.delta:+.4f}")


def demo_pipeline(model_path: str, docs: list[str]):
    """Demo: full pipeline with document loading."""
    print(f"\n{'=' * 60}")
    print("DEMO 2: OpenRAGPipeline — full RAG with quality gate")
    print("=" * 60)

    pipe = OpenRAGPipeline(model_path)
    for d in docs:
        pipe.add_file(d)
        print(f"  Loaded: {d}")

    questions = [
        "What is the main topic of these documents?",
        "What are the key findings?",
    ]

    for q in questions:
        print(f"\n  Q: {q}")
        result = pipe.query(q)
        print(f"  Result: {result}")
        if result.answer:
            print(f"  A: {result.answer[:200]}...")
        else:
            print(f"  A: [gate rejected — no confident answer]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--docs", nargs="*", default=[], help="Documents to load")
    args = parser.parse_args()

    demo_gate(args.model)
    if args.docs:
        demo_pipeline(args.model, args.docs)
