"""Multi-checkpoint entropy measurement harness.

Measures entropy at three points:
  1. Pre-retrieval: bare question entropy (trigger retrieval?)
  2. Post-retrieval: question + context entropy (context relevant?)
  3. Iterative: if score low, retrieve more and re-measure

Dynamically adjusts delta threshold based on question type
(extraction: Δ > 0.4, synthesis: Δ > 0.15).
"""
import time
from dataclasses import dataclass, field

from llama_cpp import Llama

from .entropy import measure_entropy
from .classifier import classify_question


@dataclass
class CheckpointResult:
    """Single checkpoint measurement."""
    name: str
    h_top100: float
    h_full: float
    top100_mass: float
    top5_tokens: list
    n_tokens: int
    delta_from_bare: float
    passed: bool
    context_used: str = ""  # "bare", "retrieved", "expanded"


@dataclass
class HarnessResult:
    """Full harness run on one question."""
    question: str
    question_type: str
    dynamic_threshold: float
    checkpoints: list = field(default_factory=list)
    final_verdict: str = ""  # "answer" or "abstain" or "retrieve_more"
    iterations: int = 0
    elapsed_ms: float = 0
    answer_context: str = ""


class EntropyHarness:
    """Multi-checkpoint entropy measurement with dynamic thresholds.

    Usage:
        harness = EntropyHarness("models/qwen2.5-3b-q4_k_m.gguf")
        result = harness.evaluate(
            question="Was the renovation more expensive?",
            retrieve_fn=my_retriever,
            max_iterations=3,
        )
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: int = 8,
        retrieval_entropy_threshold: float = 1.5,
        max_iterations: int = 3,
    ):
        self.retrieval_threshold = retrieval_entropy_threshold
        self.max_iterations = max_iterations
        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,
            embedding=True,
            logits_all=False,
        )

    def _checkpoint(self, text: str, name: str, bare_h: float, threshold: float) -> CheckpointResult:
        """Run a single entropy checkpoint."""
        m = measure_entropy(self._llm, text)
        delta = bare_h - m["h_top100"]
        return CheckpointResult(
            name=name,
            h_top100=m["h_top100"],
            h_full=m["h_full"],
            top100_mass=m["top100_mass"],
            top5_tokens=[(t.strip(), round(p, 3)) for t, p in m["top5_tokens"]],
            n_tokens=m["n_tokens_input"],
            delta_from_bare=delta,
            passed=delta > threshold,
        )

    def evaluate(
        self,
        question: str,
        retrieve_fn,
        top_k: int = 3,
        control_context: str | None = None,
        bare_checkpoint: CheckpointResult | None = None,
    ) -> HarnessResult:
        """Run full multi-checkpoint evaluation.

        Args:
            question: user question
            retrieve_fn: callable(question, top_k) -> list[str] of retrieved contexts
            top_k: initial retrieval count
            control_context: irrelevant context for discrimination test
            bare_checkpoint: pre-computed bare entropy (avoids redundant LLM eval)

        Returns:
            HarnessResult with all checkpoints and final verdict
        """
        start = time.monotonic()

        # Classify question → set dynamic threshold
        q_type, q_conf, threshold = classify_question(question)

        result = HarnessResult(
            question=question,
            question_type=q_type,
            dynamic_threshold=threshold,
        )

        # === CHECKPOINT 1: Bare question (use cache if available) ===
        if bare_checkpoint is not None:
            bare = bare_checkpoint
            bare_h = bare.h_top100
        else:
            bare = self._checkpoint(question, "pre_retrieval", 0.0, 0.0)
            bare_h = bare.h_top100
            bare.delta_from_bare = 0.0
            bare.passed = True
        result.checkpoints.append(bare)

        # Does the model already know? (low entropy = confident)
        if bare_h < self.retrieval_threshold:
            bare.context_used = "bare"
            result.final_verdict = "answer"
            result.answer_context = ""
            result.elapsed_ms = (time.monotonic() - start) * 1000
            return result

        # === CHECKPOINT 2: First retrieval ===
        contexts = retrieve_fn(question, top_k)
        if not contexts:
            result.final_verdict = "abstain"
            result.elapsed_ms = (time.monotonic() - start) * 1000
            return result

        combined_ctx = "\n\n".join(contexts)
        text_with_ctx = f"{combined_ctx}\n\n{question}"

        cp2 = self._checkpoint(text_with_ctx, "post_retrieval", bare_h, threshold)
        cp2.context_used = "retrieved"
        result.checkpoints.append(cp2)
        result.iterations = 1

        if cp2.passed:
            result.final_verdict = "answer"
            result.answer_context = combined_ctx
            result.elapsed_ms = (time.monotonic() - start) * 1000
            return result

        # === CHECKPOINT 3+: Iterative retrieval ===
        # Score was low — retrieve more documents and try again
        for i in range(2, self.max_iterations + 1):
            more_contexts = retrieve_fn(question, top_k * i)
            if not more_contexts:
                break

            expanded = "\n\n".join(more_contexts)
            text_expanded = f"{expanded}\n\n{question}"

            cp = self._checkpoint(
                text_expanded, f"iterative_{i}", bare_h, threshold
            )
            cp.context_used = "expanded"
            result.checkpoints.append(cp)
            result.iterations = i

            if cp.passed:
                result.final_verdict = "answer"
                result.answer_context = expanded
                result.elapsed_ms = (time.monotonic() - start) * 1000
                return result

        # Optional: control checkpoint (irrelevant context discrimination)
        if control_context:
            cp_ctrl = self._checkpoint(
                f"{control_context}\n\n{question}",
                "control_irrelevant",
                bare_h,
                threshold,
            )
            cp_ctrl.context_used = "control"
            result.checkpoints.append(cp_ctrl)

        result.final_verdict = "abstain"
        result.elapsed_ms = (time.monotonic() - start) * 1000
        return result

    def evaluate_batch(
        self,
        questions: list[str],
        retrieve_fn,
        top_k: int = 3,
    ) -> list[HarnessResult]:
        """Run harness on multiple questions."""
        return [self.evaluate(q, retrieve_fn, top_k) for q in questions]

    def report(self, results: list[HarnessResult]) -> str:
        """Generate text report from batch results."""
        lines = []
        lines.append("=" * 70)
        lines.append("ENTROPY HARNESS REPORT")
        lines.append("=" * 70)

        # Per-question summary
        lines.append(f"\n  {'Q#':>3} {'Type':<12} {'Thresh':>6} {'Iters':>5} {'Verdict':<14} {'Bare':>7} {'Delta':>7}")
        lines.append(f"  {'-' * 60}")

        total = len(results)
        extraction_results = [r for r in results if r.question_type == "extraction"]
        synthesis_results = [r for r in results if r.question_type == "synthesis"]

        for i, r in enumerate(results):
            bare_h = r.checkpoints[0].h_top100 if r.checkpoints else 0
            best_delta = max(
                (cp.delta_from_bare for cp in r.checkpoints[1:]),
                default=0,
            )
            lines.append(
                f"  {i+1:>3} {r.question_type:<12} {r.dynamic_threshold:>6.2f} "
                f"{r.iterations:>5} {r.final_verdict:<14} {bare_h:>7.4f} {best_delta:>+7.4f}"
            )

        answered = sum(1 for r in results if r.final_verdict == "answer")
        abstained = sum(1 for r in results if r.final_verdict == "abstain")

        lines.append(f"\n  Total: {total}  Answered: {answered}  Abstained: {abstained}")

        if extraction_results:
            ext_answered = sum(1 for r in extraction_results if r.final_verdict == "answer")
            lines.append(f"  Extraction: {len(extraction_results)}  Answered: {ext_answered}")

        if synthesis_results:
            syn_answered = sum(1 for r in synthesis_results if r.final_verdict == "answer")
            lines.append(f"  Synthesis:  {len(synthesis_results)}  Answered: {syn_answered}")

        return "\n".join(lines)
