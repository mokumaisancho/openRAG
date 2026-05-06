"""openRAG — Entropy-based RAG quality gate."""
from .entropy import measure_entropy, SignalResult
from .gate import EntropyGate
from .retriever import TFIDFRetriever
from .pipeline import OpenRAGPipeline, QueryResult

__version__ = "0.1.0"
