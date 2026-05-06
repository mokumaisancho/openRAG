"""openRAG — Entropy-based RAG quality gate."""
from .entropy import measure_entropy, SignalResult
from .gate import EntropyGate
from .retriever import TFIDFRetriever
from .pipeline import OpenRAGPipeline, QueryResult
from .classifier import classify_question
from .harness import EntropyHarness, HarnessResult

__all__ = [
    "measure_entropy", "SignalResult",
    "EntropyGate",
    "TFIDFRetriever",
    "OpenRAGPipeline", "QueryResult",
    "classify_question",
    "EntropyHarness", "HarnessResult",
]
__version__ = "0.1.0"
