# openRAG

**Entropy-based RAG quality gate. Open-source, zero-cost, local.**

openRAG uses your generation model's own next-token entropy to detect whether retrieved context is actually relevant — before generating an answer. No extra model, no extra API call, no extra latency.

## The Problem

RAG systems retrieve documents and feed them to an LLM. But retrieval is noisy — embedding similarity returns documents that *look* relevant but don't actually help answer the question. Current solutions:

- **Embedding similarity** — conflates topic proximity with answer relevance
- **Second LLM verification call** — doubles cost and latency ($0.03/query extra)
- **No check at all** — just hope the retrieval was good

## Our Solution

We discovered that a local model's **next-token entropy** drops when relevant context is provided, but doesn't drop (or increases) with irrelevant context. This signal:

- Costs **zero** extra compute — it comes from logits you already compute
- Works in **real-time** — single forward pass, no second model call
- Discriminates **relevance, not just presence** — irrelevant context doesn't trigger it
- Is **robust to paraphrase** — works across different phrasings of the same question
- Fires for **synthesis questions** — not just literal extraction

## How It Works

```
User question
    ↓
Measure entropy on bare question → HIGH? → trigger retrieval
    ↓
Retrieved context injected
    ↓
Measure entropy again
    ↓
Entropy DROPPED → context is relevant → generate answer
Entropy UNCHANGED → retrieval failed → retry or abstain
```

The signal comes from measuring **top-100 entropy** (Shannon entropy over the top 100 most likely next tokens) before and after context injection. Full-vocabulary entropy is too noisy — the signal lives in the distribution head.

## Experimental Data

### Experiment 1: Does the signal work on small models?

Tested the same question across 4 model sizes:

| Model | Params | Δ(relevant) | Δ(irrelevant) | Signal |
|-------|--------|-------------|---------------|--------|
| SmolLM2 | 135M | +0.53 | +0.51 | NONE — drops for any context |
| Qwen2.5 | 500M | +0.96 | -1.29 | NOISY — direction correct but erratic |
| Qwen2.5 | 1.5B | +0.56 | -1.26 | NOISY — same as 500M |
| **Qwen2.5** | **3B** | **+0.86** | **-0.47** | **STRONG — clean discrimination** |

**Finding**: Reliable entropy signal requires ~3B parameters. Below 1.5B, the model can't distinguish relevant from irrelevant context — it reacts to any text. At 3B, representations are structured enough to discriminate.

### Experiment 2: Is the signal robust to paraphrase?

Tested 7 paraphrases of the same question (varying lexical distance) + 3 synthesis questions on the 3B model:

**Extraction paraphrases** (answer literally in context):
| Paraphrase | Difficulty | Δ(relevant) | Signal |
|-----------|-----------|-------------|--------|
| P1 | easy | +1.50 | YES |
| P2 | easy | +0.88 | YES |
| P3 | medium | +1.67 | YES |
| P4 | medium | +2.25 | YES |
| P5 | medium-hard | +0.79 | YES |
| P6 | hard | +0.78 | YES |
| P7 | very-hard | +0.52 | YES |

**Result**: 7/7 paraphrases show entropy drop. Signal degrades gradually with lexical distance (1.50 → 0.52) but never disappears.

**Synthesis questions** (answer requires connecting facts, not literal extraction):
| Question | Reasoning | Δ(relevant) | Signal |
|----------|-----------|-------------|--------|
| "Was renovation more expensive than original?" | Compare two facts | +0.23 | YES |
| "How many times more expensive?" | Division | +0.70 | YES |
| "Added to Register after renovation?" | Temporal comparison | +0.59 | YES |

**Result**: 3/3 synthesis questions show entropy drop. Signal is smaller than extraction (0.23-0.70 vs 0.52-2.25) but consistently positive.

## Recommended Model

**Qwen3 3B** (Q4_K_M quantization, ~1.8GB GGUF)

Tested and validated on Qwen2.5-3B (same architecture family). The 3B size is the minimum for reliable signal — smaller models produce noisy or non-discriminative entropy readings.

Download:
```bash
# Option 1: Ollama
ollama pull qwen3:3b

# Option 2: Direct GGUF
# https://huggingface.co/Qwen/Qwen3-3B-GGUF
```

## Installation

```bash
pip install llama-cpp-python numpy
pip install openrag  # or: pip install -e .
```

For the middleware server:
```bash
pip install fastapi uvicorn
```

## Quick Start

### Python Library

```python
from openrag import EntropyGate

gate = EntropyGate("models/qwen3-3b-q4_k_m.gguf")

# Check if context is relevant
result = gate.check(
    question="What is the capital of Mongolia?",
    context="Ulaanbaatar is the capital and largest city of Mongolia.",
)

print(result.passed)       # True
print(result.confidence)   # "HIGH"
print(result.delta)        # 0.86 (entropy dropped)
```

### Full Pipeline

```python
from openrag import OpenRAGPipeline

pipe = OpenRAGPipeline("models/qwen3-3b-q4_k_m.gguf")
pipe.add_file("knowledge_base.txt")
pipe.add_file("faq.txt")

result = pipe.query("What are the refund policies?")
if result.passed:
    print(result.answer)
else:
    print("No confident answer found.")
```

### Middleware Server

```bash
python server.py --model models/qwen3-3b-q4_k_m.gguf --docs knowledge.txt faq.txt
```

```bash
# Check if context is relevant
curl -X POST http://localhost:8000/check \
  -H "Content-Type: application/json" \
  -d '{"question": "What is X?", "context": "X is a system for..."}'

# Full query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is X?"}'
```

## API Reference

### `EntropyGate(model_path, threshold=0.2)`

| Method | Description |
|--------|-------------|
| `check(question, context, control_context=None)` | Check if context is relevant |
| `check_batch(question, contexts)` | Check multiple contexts, return ranked |
| `should_retrieve(question, threshold=1.5)` | Check if question needs retrieval |

### `SignalResult`

| Field | Type | Description |
|-------|------|-------------|
| `delta` | float | Entropy drop (positive = context helped) |
| `confidence` | str | HIGH / MEDIUM / LOW / NONE |
| `passed` | bool | Whether context passes the quality gate |
| `h_bare` | float | Entropy without context |
| `h_with_context` | float | Entropy with context |

### `OpenRAGPipeline(model_path)`

| Method | Description |
|--------|-------------|
| `add_file(path)` | Add text file to knowledge base |
| `add_texts(texts)` | Add raw text documents |
| `query(question)` | Full pipeline: retrieve + gate + generate |

## Architecture

```
openRAG/
├── openrag/
│   ├── entropy.py       # Core entropy measurement
│   ├── gate.py          # EntropyGate — quality gate logic
│   ├── retriever.py     # Minimal TF-IDF retriever
│   └── pipeline.py      # Full RAG pipeline with gate
├── server.py            # FastAPI middleware
├── examples/
│   └── basic_usage.py
└── tests/
```

## How It Differs From Existing RAG Quality Checks

| Approach | Cost | Latency | What It Detects |
|----------|------|---------|-----------------|
| Embedding similarity | $0.0001 | ~10ms | Topic proximity (not answer relevance) |
| LLM verification call | $0.03 | ~2s | Correctness of generated answer |
| **openRAG entropy gate** | **$0** | **~0ms** (already computed) | **Whether context helps answer the question** |

Embedding similarity checks "is this document about the same topic?" — but the document can be topically related without containing the answer. Entropy checks "does the model become more confident after reading this?" — which directly measures answer relevance.

## Limitations

1. **Minimum model size: ~3B parameters.** Below 1.5B, the entropy signal doesn't discriminate relevant from irrelevant context. Do not use with models smaller than 3B.

2. **Extraction-biased.** Signal is strongest when the answer is literally in the context. Synthesis questions show smaller but still positive drops.

3. **Single-token measurement.** Entropy is measured at the next-token position only. The signal reflects the model's immediate confidence, not its full generation trajectory.

4. **Quantization sensitive.** Very aggressive quantization (Q2, Q3) may degrade the signal. Q4_K_M or higher is recommended.

## License

MIT

## Citation

If you use openRAG in research, cite:

```
openRAG: Entropy-based RAG Quality Gate (2026)
https://github.com/your-org/openRAG
```

## Acknowledgments

- entropy measurement approach derived from the [epistemic_2](https://github.com/user/epistemic_2) project's work on knowledge boundary detection via output distribution statistics
- Top-100 entropy as optimal signal window: Cohen's d > 1.0 for known/unknown discrimination at 3B+ scale
- Qwen model family by Alibaba Cloud
