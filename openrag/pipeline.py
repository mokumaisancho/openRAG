"""openRAG pipeline — retrieval + entropy quality gate + generation."""
from .gate import EntropyGate
from .retriever import TFIDFRetriever
from .entropy import SignalResult


class OpenRAGPipeline:
    """Full RAG pipeline with entropy-based quality gating.

    1. Receive question
    2. Check if retrieval is needed (should_retrieve)
    3. Retrieve candidate documents
    4. Quality-gate each document (entropy check)
    5. Generate answer using only gate-approved context

    Usage:
        pipe = OpenRAGPipeline("models/qwen2.5-3b-q4_k_m.gguf")
        pipe.add_documents(["doc1.txt", "doc2.txt"])
        result = pipe.query("What is X?")
        print(result.answer if result.passed else "I don't know")
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: int = 8,
        gate_threshold: float = 0.2,
        retrieval_entropy_threshold: float = 1.5,
    ):
        self.gate = EntropyGate(model_path, n_ctx, n_threads, gate_threshold)
        self.retriever = TFIDFRetriever()
        self.retrieval_threshold = retrieval_entropy_threshold

    def add_texts(self, texts: list[str], metadatas: list[dict] | None = None):
        """Add text documents to the knowledge base."""
        self.retriever.add_texts(texts, metadatas)

    def add_file(self, path: str):
        """Add a text file to the knowledge base."""
        self.retriever.add_file(path)

    def add_files(self, paths: list[str]):
        """Add multiple text files."""
        for p in paths:
            self.add_file(p)

    def query(
        self,
        question: str,
        top_k: int = 3,
        control_context: str | None = None,
    ) -> "QueryResult":
        """Query the pipeline.

        Args:
            question: user question
            top_k: number of documents to retrieve
            control_context: irrelevant context for discrimination check

        Returns:
            QueryResult with answer, signal info, and gate status
        """
        # Step 1: Check if retrieval is needed
        needs_retrieval = self.gate.should_retrieve(question, self.retrieval_threshold)

        if not needs_retrieval:
            bare = self.gate._llm.create_chat_completion(
                messages=[{"role": "user", "content": question}],
                max_tokens=256,
            )
            return QueryResult(
                question=question,
                answer=bare["choices"][0]["message"]["content"],
                passed=True,
                confidence="HIGH",
                signal=None,
                source=None,
                retrieval_needed=False,
            )

        # Step 2: Retrieve candidates
        candidates = self.retriever.retrieve(question, top_k)
        if not candidates:
            return QueryResult(
                question=question,
                answer=None,
                passed=False,
                confidence="NONE",
                signal=None,
                source=None,
                retrieval_needed=True,
            )

        # Step 3: Quality-gate each candidate
        contexts = [doc.text for doc, _ in candidates]
        ranked = self.gate.check_batch(question, contexts, control_context)

        best_idx, best_signal = ranked[0]

        if best_signal.passed:
            best_doc = candidates[best_idx][0]
            ctx_text = best_doc.text

            response = self.gate._llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": f"Answer based on this context:\n{ctx_text}"},
                    {"role": "user", "content": question},
                ],
                max_tokens=256,
            )
            answer = response["choices"][0]["message"]["content"]
        else:
            answer = None

        return QueryResult(
            question=question,
            answer=answer,
            passed=best_signal.passed,
            confidence=best_signal.confidence,
            signal=best_signal,
            source=candidates[best_idx][0].metadata if best_signal.passed else None,
            retrieval_needed=True,
        )


class QueryResult:
    """Result of a RAG query."""

    def __init__(
        self,
        question: str,
        answer: str | None,
        passed: bool,
        confidence: str,
        signal: SignalResult | None,
        source: dict | None,
        retrieval_needed: bool,
    ):
        self.question = question
        self.answer = answer
        self.passed = passed
        self.confidence = confidence
        self.signal = signal
        self.source = source
        self.retrieval_needed = retrieval_needed

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"QueryResult({status}, confidence={self.confidence}, retrieval={'yes' if self.retrieval_needed else 'no'})"

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "passed": self.passed,
            "confidence": self.confidence,
            "retrieval_needed": self.retrieval_needed,
            "source": self.source,
            "signal": {
                "h_bare": self.signal.h_bare,
                "h_with_context": self.signal.h_with_context,
                "delta": self.signal.delta,
                "confidence": self.signal.confidence,
            } if self.signal else None,
        }
