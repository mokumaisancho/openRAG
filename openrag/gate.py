"""EntropyGate — checks if retrieved context actually helps answer a question."""
from llama_cpp import Llama
from .entropy import measure_entropy, classify_signal, SignalResult


class EntropyGate:
    """Quality gate for RAG retrieval using model entropy.

    Loads a local GGUF model and measures whether retrieved context
    reduces the model's next-token entropy — indicating the context
    is relevant and helps answer the question.

    Usage:
        gate = EntropyGate("models/qwen2.5-3b-q4_k_m.gguf")
        result = gate.check("What is X?", "Context about X...")
        if result.passed:
            # context is relevant, proceed to generation
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: int = 8,
        threshold: float = 0.2,
    ):
        """
        Args:
            model_path: path to GGUF model file
            n_ctx: context window size
            n_threads: number of CPU threads
            threshold: minimum entropy drop to pass (nat)
        """
        self.threshold = threshold
        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,
            embedding=True,
            logits_all=True,
        )

    def check(
        self,
        question: str,
        context: str,
        control_context: str | None = None,
    ) -> SignalResult:
        """Check if context is relevant for answering the question.

        Args:
            question: user question
            context: retrieved context to validate
            control_context: irrelevant context for discrimination check

        Returns:
            SignalResult with confidence, pass/fail, and metrics
        """
        bare = measure_entropy(self._llm, question)
        with_ctx = measure_entropy(self._llm, f"{context}\n\n{question}")

        delta = bare["h_top100"] - with_ctx["h_top100"]

        h_control = None
        delta_control = None
        if control_context:
            ctrl = measure_entropy(self._llm, f"{control_context}\n\n{question}")
            h_control = ctrl["h_top100"]
            delta_control = bare["h_top100"] - ctrl["h_top100"]

        confidence, passed = classify_signal(delta, delta_control)

        return SignalResult(
            h_bare=bare["h_top100"],
            h_with_context=with_ctx["h_top100"],
            h_control=h_control,
            delta=delta,
            confidence=confidence,
            passed=passed,
            top5_bare=[(t.strip(), round(p, 3)) for t, p in bare["top5_tokens"]],
            top5_context=[(t.strip(), round(p, 3)) for t, p in with_ctx["top5_tokens"]],
            n_tokens_input=with_ctx["n_tokens_input"],
        )

    def check_batch(
        self,
        question: str,
        contexts: list[str],
        control_context: str | None = None,
    ) -> list[tuple[int, SignalResult]]:
        """Check multiple contexts, return ranked by entropy drop.

        Returns:
            list of (index, SignalResult) sorted by delta descending
        """
        results = []
        for i, ctx in enumerate(contexts):
            result = self.check(question, ctx, control_context)
            results.append((i, result))

        results.sort(key=lambda x: x[1].delta, reverse=True)
        return results

    def should_retrieve(self, question: str, threshold: float | None = None) -> bool:
        """Check if the question needs retrieval at all.

        If entropy on the bare question is below threshold, the model
        already knows the answer and retrieval is unnecessary.

        Args:
            question: user question
            threshold: entropy threshold for triggering retrieval

        Returns:
            True if retrieval is recommended
        """
        thresh = threshold or 1.5
        bare = measure_entropy(self._llm, question)
        return bare["h_top100"] > thresh
