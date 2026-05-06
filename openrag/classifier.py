"""Question type classifier — extraction vs synthesis detection.

Uses keyword heuristics to classify questions before setting the
entropy gate threshold. Extraction questions get a higher bar,
synthesis questions get a lower bar.
"""
import re

SYNTHESIS_PATTERNS = [
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


def classify_question(question: str) -> dict:
    """Classify a question as extraction or synthesis.

    Returns:
        dict with:
            type: "extraction" or "synthesis"
            confidence: "high" or "low"
            matched_patterns: list of matched synthesis patterns
            threshold: recommended entropy delta threshold
    """
    q_lower = question.lower().strip()

    matched = []
    for pat in SYNTHESIS_PATTERNS:
        if re.search(pat, q_lower):
            matched.append(pat)

    is_synthesis = len(matched) >= 1

    # Questions with multiple synthesis markers are "high confidence" synthesis
    synthesis_confidence = "high" if len(matched) >= 2 else "low"

    if is_synthesis:
        q_type = "synthesis"
        # Lower threshold for synthesis — our data shows delta 0.23-0.70
        threshold = 0.15
    else:
        q_type = "extraction"
        # Higher threshold for extraction — our data shows delta 0.52-2.25
        threshold = 0.4

    return {
        "type": q_type,
        "confidence": synthesis_confidence if is_synthesis else "high",
        "matched_patterns": matched,
        "threshold": threshold,
    }
