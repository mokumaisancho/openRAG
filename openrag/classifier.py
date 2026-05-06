"""Question type classifier — extraction vs synthesis detection.

Uses keyword heuristics to classify questions before setting the
entropy gate threshold. Extraction questions get a higher bar,
synthesis questions get a lower bar.
"""
import re
from functools import lru_cache

_SYNTHESIS_PATTERNS = [
    re.compile(p) for p in [
        r"\bmore\s+than\b",
        r"\bless\s+than\b",
        r"\bcompared?\b",
        r"\bcomparison\b",
        r"\bversus\b",
        r"\bvs\.?\b",
        r"\bdiffer\b",
        r"\bdifference\b",
        r"\bhow\s+many\s+times\b",
        r"\bwhat\s+fraction\b",
        r"\bwhat\s+percentage\b",
        r"\bratio\b",
        r"\bmultipl[ey]\b",
        r"\bbefore\b.*\bafter\b",
        r"\bafter\b.*\bbefore\b",
        r"\btrue\s+or\s+false\b",
        r"\bis\s+it\s+true\b",
        r"\bcorrect\s+or\s+incorrect\b",
        r"\bwhy\s+did\b",
        r"\bwhy\s+is\b",
        r"\bcause\b",
        r"\breason\b",
        r"\bconsequence\b",
        r"\bresult\s+of\b",
        r"\bimpact\s+of\b",
        r"\beffect\s+of\b",
        r"\brelationship\s+between\b",
        r"\bbetter\b",
        r"\bworse\b",
        r"\bsuperior\b",
        r"\binferior\b",
        r"\bmost\s+\w+\b",
        r"\bleast\s+\w+\b",
    ]
]


@lru_cache(maxsize=512)
def classify_question(question: str) -> tuple:
    """Classify a question as extraction or synthesis.

    Returns:
        tuple of (type, confidence, threshold)
    """
    q_lower = question.lower().strip()

    n_matched = sum(1 for p in _SYNTHESIS_PATTERNS if p.search(q_lower))

    if n_matched >= 1:
        confidence = "high" if n_matched >= 2 else "low"
        return ("synthesis", confidence, 0.15)
    else:
        return ("extraction", "high", 0.4)
