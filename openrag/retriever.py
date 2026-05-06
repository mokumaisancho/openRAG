"""Minimal TF-IDF retriever — no external embedding model needed."""
import math
import re
from dataclasses import dataclass


@dataclass
class Document:
    text: str
    metadata: dict | None = None
    _tokens: list[str] | None = None

    @property
    def tokens(self) -> list[str]:
        if self._tokens is None:
            self._tokens = re.findall(r"\w+", self.text.lower())
        return self._tokens


class TFIDFRetriever:
    """Simple TF-IDF retriever for demonstration and lightweight use.

    For production use, replace with an embedding-based retriever
    (sentence-transformers, OpenAI embeddings, etc.). This implementation
    exists so openRAG works out-of-the-box with zero external dependencies
    beyond llama-cpp-python and numpy.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents: list[Document] = []
        self._idf: dict[str, float] = {}
        self._fitted = False

    def add_texts(self, texts: list[str], metadatas: list[dict] | None = None):
        """Add text documents to the retriever."""
        start = len(self.documents)
        for i, text in enumerate(texts):
            chunks = self._chunk(text)
            meta = metadatas[i] if metadatas and i < len(metadatas) else {}
            for chunk in chunks:
                self.documents.append(Document(text=chunk, metadata={**meta, "source_index": start + i}))

        self._fit()

    def add_file(self, path: str):
        """Add a text file to the retriever."""
        with open(path) as f:
            text = f.read()
        self.add_texts([text], [{"source": path}])

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[Document, float]]:
        """Retrieve top-k documents ranked by TF-IDF cosine similarity.

        Returns:
            list of (Document, score) tuples, highest score first
        """
        if not self._fitted:
            return []

        query_tokens = re.findall(r"\w+", query.lower())
        query_vec = self._tfidf_vector(query_tokens)

        scores = []
        for doc in self.documents:
            doc_vec = self._tfidf_vector(doc.tokens)
            sim = self._cosine(query_vec, doc_vec)
            scores.append((doc, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _chunk(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        if len(words) <= self.chunk_size:
            return [text]

        chunks = []
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = " ".join(words[i : i + self.chunk_size])
            chunks.append(chunk)
            if i + self.chunk_size >= len(words):
                break
        return chunks

    def _fit(self):
        """Compute IDF across all documents."""
        n_docs = len(self.documents)
        df: dict[str, int] = {}
        for doc in self.documents:
            seen = set(doc.tokens)
            for t in seen:
                df[t] = df.get(t, 0) + 1

        self._idf = {t: math.log(n_docs / (count + 1)) + 1 for t, count in df.items()}
        self._fitted = True

    def _tfidf_vector(self, tokens: list[str]) -> dict[str, float]:
        """Compute TF-IDF vector as sparse dict."""
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1

        vec = {}
        for t, count in tf.items():
            if t in self._idf:
                vec[t] = (count / len(tokens)) * self._idf[t]
        return vec

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        """Cosine similarity between two sparse vectors."""
        common = set(a.keys()) & set(b.keys())
        if not common:
            return 0.0
        dot = sum(a[k] * b[k] for k in common)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
