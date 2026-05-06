"""
openRAG — Entropy-based RAG quality gate.

Measures whether retrieved context actually helps the model answer a question.
Uses next-token entropy from the generation model's own logits — no extra model needed.
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class SignalResult:
    """Result of entropy-based quality check."""
    h_bare: float          # Top-100 entropy without context
    h_with_context: float  # Top-100 entropy with context
    h_control: float       # Top-100 entropy with irrelevant context (optional)
    delta: float           # h_bare - h_with_context (positive = entropy dropped)
    confidence: str        # HIGH / MEDIUM / LOW / NONE
    passed: bool           # True if context is relevant enough
    top5_bare: list        # Top-5 tokens without context
    top5_context: list     # Top-5 tokens with context
    n_tokens_input: int    # Number of input tokens


def measure_entropy(llm, text: str) -> dict:
    """Measure entropy from full vocabulary logits at last token position.

    Args:
        llm: llama_cpp.Llama instance (must be loaded with logits_all=True)
        text: input text to evaluate

    Returns:
        dict with h_full, h_top100, top100_mass, top5_tokens, n_tokens_input
    """
    llm.reset()
    tokens = llm.tokenize(text.encode())
    llm.eval(tokens)
    logits = llm._scores[-1].copy()

    logits_max = np.max(logits)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / np.sum(exp_logits)

    log_probs = np.log(probs + 1e-10)
    h_full = float(-np.sum(probs * log_probs))

    top_k = 100
    top_indices = np.argpartition(logits, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(logits[top_indices])[::-1]]
    top_probs = probs[top_indices]
    top100_mass = float(np.sum(top_probs))
    top_probs_norm = top_probs / top100_mass
    h_top100 = float(-np.sum(top_probs_norm * np.log(top_probs_norm + 1e-10)))

    top5_indices = np.argsort(logits)[-5:][::-1]
    top5_tokens = [llm.detokenize([int(i)]).decode(errors="replace") for i in top5_indices]
    top5_probs = [float(probs[i]) for i in top5_indices]

    return {
        "h_full": h_full,
        "h_top100": h_top100,
        "top100_mass": top100_mass,
        "top5_tokens": list(zip(top5_tokens, top5_probs)),
        "n_tokens_input": len(tokens),
    }


def classify_signal(delta: float, delta_control: float | None = None) -> tuple[str, bool]:
    """Classify signal strength and whether it passes the quality gate.

    Args:
        delta: entropy drop (h_bare - h_with_context)
        delta_control: entropy change with irrelevant context (optional)

    Returns:
        (confidence_level, passed)
    """
    if delta_control is not None:
        # With control: check discrimination
        if delta > 0.5 and delta > abs(delta_control) * 1.5:
            return "HIGH", True
        elif delta > 0.2 and delta > abs(delta_control):
            return "MEDIUM", True
        elif delta > 0.1:
            return "LOW", False
        else:
            return "NONE", False
    else:
        # Without control: only use delta magnitude
        if delta > 0.5:
            return "HIGH", True
        elif delta > 0.2:
            return "MEDIUM", True
        elif delta > 0.1:
            return "LOW", False
        else:
            return "NONE", False
