"""RGB Benchmark runner for openRAG entropy quality gate.

Downloads and runs the RGB (Retrieval-Augmented Generation Benchmark)
by Chen et al., AAAI 2024. Tests 4 RAG abilities:
  1. Noise Robustness — answer correctly with noisy documents
  2. Negative Rejection — refuse to answer when no relevant docs exist
  3. Information Integration — combine info from multiple docs
  4. Counterfactual Robustness — detect docs with fake facts

The entropy harness measures at each checkpoint:
  - Pre-retrieval: bare question entropy
  - Post-retrieval: entropy after injecting context
  - Dynamic threshold: extraction (Δ > 0.4) vs synthesis (Δ > 0.15)

Usage:
    python benchmark_rgb.py --model /path/to/model.gguf

RGB baseline scores to beat (ChatGPT / best open-source):
  Noise Robustness (noise=0.6):  88-89% accuracy
  Negative Rejection (Rej*):     45% (ChatGPT is best)
  Information Integration:       55-67% accuracy
  Counterfactual Error Detect:    3-8% (all models are terrible)
"""
import argparse
import json
import os
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from openrag.entropy import measure_entropy
from openrag.harness import EntropyHarness, CheckpointResult

SCRIPT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(SCRIPT_DIR, "benchmark_data", "rgb")

RGB_URLS = {
    "en": "https://raw.githubusercontent.com/chen700564/RGB/master/data/en.json",
    "en_int": "https://raw.githubusercontent.com/chen700564/RGB/master/data/en_int.json",
    "en_fact": "https://raw.githubusercontent.com/chen700564/RGB/master/data/en_fact.json",
}

# Number of positive/negative passages per question (RGB default: 5)
PASSAGE_NUM = 5
NOISE_RATIOS = [0.0, 0.2, 0.4, 0.6]


def download_rgb():
    """Download RGB benchmark data."""
    os.makedirs(DATA_DIR, exist_ok=True)
    for name, url in RGB_URLS.items():
        path = os.path.join(DATA_DIR, f"{name}.json")
        if os.path.exists(path):
            continue
        print(f"  Downloading {name}...")
        urllib.request.urlretrieve(url, path)
    print("  RGB data ready.")


def load_rgb():
    """Load RGB benchmark data (JSONL format)."""
    data = {}
    for name in RGB_URLS:
        path = os.path.join(DATA_DIR, f"{name}.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            data[name] = [json.loads(line) for line in f if line.strip()]
    return data


# ─── Evaluation Functions ──────────────────────────────────────────


def precompute_bare_entropy(harness, samples) -> dict:
    """Cache bare-question entropy once per unique query.

    Avoids redundant LLM evals across noise ratios and test types.
    """
    cache = {}
    for sample in samples:
        query = sample["query"]
        if query not in cache:
            m = measure_entropy(harness._llm, query)
            cp = CheckpointResult(
                name="pre_retrieval",
                h_top100=m["h_top100"],
                h_full=m["h_full"],
                top100_mass=m["top100_mass"],
                top5_tokens=[(t, round(p, 3)) for t, p in m["top5_tokens"]],
                n_tokens=m["n_tokens_input"],
                delta_from_bare=0.0,
                passed=True,
                context_used="bare",
            )
            cache[query] = cp
    return cache


def run_noise_robustness(harness, samples, noise_ratio: float, bare_cache: dict) -> dict:
    """Test noise robustness: answer correctly with noisy documents mixed in.

    Args:
        harness: EntropyHarness instance
        samples: list of RGB samples (en.json)
        noise_ratio: fraction of passages that are negative (irrelevant)

    Returns:
        dict with accuracy, per-sample results
    """
    results = []
    gate_passed = 0
    total = 0

    n_noise = int(PASSAGE_NUM * noise_ratio)
    n_positive = PASSAGE_NUM - n_noise

    for sample in samples:
        query = sample["query"]

        # Build context: mix positive and negative passages
        positives = sample.get("positive", [])[:n_positive]
        negatives = sample.get("negative", [])[:n_noise]

        if not positives and noise_ratio == 0:
            continue

        context_passages = positives + negatives
        # Shuffle to avoid position bias
        np.random.shuffle(context_passages)
        context = "\n\n".join(context_passages)

        def retrieve_fn(q, k):
            return [context]

        result = harness.evaluate(query, retrieve_fn, top_k=1, bare_checkpoint=bare_cache.get(query))
        results.append({
            "id": sample["id"],
            "query": query,
            "question_type": result.question_type,
            "threshold": result.dynamic_threshold,
            "verdict": result.final_verdict,
            "iterations": result.iterations,
            "delta": result.checkpoints[1].delta_from_bare if len(result.checkpoints) > 1 else 0,
            "bare_h": result.checkpoints[0].h_top100,
        })

        total += 1
        if result.final_verdict == "answer":
            gate_passed += 1

    return {
        "noise_ratio": noise_ratio,
        "total": total,
        "gate_passed": gate_passed,
        "gate_pass_rate": gate_passed / max(total, 1),
        "results": results,
    }


def run_negative_rejection(harness, samples, bare_cache: dict) -> dict:
    """Test negative rejection: refuse to answer when only irrelevant docs exist.

    Uses only negative passages (no positives). The entropy gate should
    detect that context doesn't help and abstain.
    """
    results = []
    rejected = 0
    total = 0

    for sample in samples:
        query = sample["query"]
        negatives = sample.get("negative", [])[:PASSAGE_NUM]

        if not negatives:
            continue

        context = "\n\n".join(negatives)

        def retrieve_fn(q, k):
            return [context]

        result = harness.evaluate(query, retrieve_fn, top_k=1, bare_checkpoint=bare_cache.get(query))

        total += 1
        if result.final_verdict == "abstain":
            rejected += 1

        results.append({
            "id": sample["id"],
            "query": query,
            "verdict": result.final_verdict,
            "delta": result.checkpoints[1].delta_from_bare if len(result.checkpoints) > 1 else 0,
            "bare_h": result.checkpoints[0].h_top100,
        })

    return {
        "total": total,
        "rejected": rejected,
        "rejection_rate": rejected / max(total, 1),
        "results": results,
    }


def run_information_integration(harness, samples, noise_ratio: float) -> dict:
    """Test information integration: combine facts from multiple docs."""
    results = []
    gate_passed = 0
    total = 0

    for sample in samples:
        query = sample["query"]

        # Integration samples have positive as list of lists
        positive_groups = sample.get("positive", [])
        negatives = sample.get("negative", [])

        n_noise = int(len(positive_groups) * noise_ratio)
        context_passages = []

        for group in positive_groups:
            if isinstance(group, list):
                context_passages.extend(group[:1])
            else:
                context_passages.append(group)

        context_passages.extend(negatives[:n_noise])
        context = "\n\n".join(context_passages)

        def retrieve_fn(q, k):
            return [context]

        result = harness.evaluate(query, retrieve_fn, top_k=1)

        total += 1
        if result.final_verdict == "answer":
            gate_passed += 1

        results.append({
            "id": sample["id"],
            "query": query,
            "question_type": result.question_type,
            "verdict": result.final_verdict,
            "delta": result.checkpoints[1].delta_from_bare if len(result.checkpoints) > 1 else 0,
        })

    return {
        "noise_ratio": noise_ratio,
        "total": total,
        "gate_passed": gate_passed,
        "gate_pass_rate": gate_passed / max(total, 1),
        "results": results,
    }


def run_counterfactual(harness, samples) -> dict:
    """Test counterfactual robustness: detect documents with fake facts.

    Uses positive_wrong (docs with fake answer). The entropy gate
    should detect that something is off and flag it.
    """
    results = []
    total = 0

    for sample in samples:
        query = sample["query"]
        wrong_docs = sample.get("positive_wrong", [])

        if not wrong_docs:
            continue

        context = "\n\n".join(wrong_docs[:PASSAGE_NUM])

        def retrieve_fn(q, k):
            return [context]

        result = harness.evaluate(query, retrieve_fn, top_k=1)

        total += 1
        delta = result.checkpoints[1].delta_from_bare if len(result.checkpoints) > 1 else 0

        results.append({
            "id": sample["id"],
            "query": query,
            "fake_answer": sample.get("fakeanswer", ""),
            "verdict": result.final_verdict,
            "delta": delta,
            "bare_h": result.checkpoints[0].h_top100,
        })

    return {
        "total": total,
        "results": results,
    }


# ─── Main ──────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="RGB benchmark for openRAG")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--samples", type=int, default=50, help="Samples per test (default: 50)")
    parser.add_argument("--output", default="results/rgb_benchmark.json")
    args = parser.parse_args()

    download_rgb()
    data = load_rgb()
    if not data:
        print("Failed to load RGB data.")
        sys.exit(1)

    print(f"\nLoading model: {args.model}")
    harness = EntropyHarness(args.model)

    all_results = {}
    n = args.samples

    # ─── Test 1: Noise Robustness ───
    print(f"\n{'=' * 60}")
    print("TEST 1: Noise Robustness")
    print(f"{'=' * 60}")
    en_samples = data["en"][:n]

    # Pre-compute bare entropy once (saves 150+ redundant LLM evals)
    print(f"  Pre-computing bare entropy for {len(en_samples)} questions...", flush=True)
    bare_cache = precompute_bare_entropy(harness, en_samples)
    print("  Done.")

    noise_results = {}
    for ratio in NOISE_RATIOS:
        print(f"  Noise={ratio:.1f} ...", end=" ", flush=True)
        r = run_noise_robustness(harness, en_samples, ratio, bare_cache)
        noise_results[str(ratio)] = r
        print(f"gate_pass={r['gate_pass_rate']:.1%}")

    all_results["noise_robustness"] = noise_results

    # ─── Test 2: Negative Rejection ───
    print(f"\n{'=' * 60}")
    print("TEST 2: Negative Rejection")
    print(f"{'=' * 60}")
    print(f"  Running {n} samples...", end=" ", flush=True)
    neg_result = run_negative_rejection(harness, en_samples, bare_cache)
    all_results["negative_rejection"] = neg_result
    print(f"rejection_rate={neg_result['rejection_rate']:.1%}")

    # ─── Test 3: Information Integration ───
    print(f"\n{'=' * 60}")
    print("TEST 3: Information Integration")
    print(f"{'=' * 60}")
    int_samples = data.get("en_int", [])[:n]

    int_results = {}
    for ratio in [0.0, 0.2, 0.4]:
        if int_samples:
            print(f"  Noise={ratio:.1f} ...", end=" ", flush=True)
            r = run_information_integration(harness, int_samples, ratio)
            int_results[str(ratio)] = r
            print(f"gate_pass={r['gate_pass_rate']:.1%}")
    all_results["information_integration"] = int_results

    # ─── Test 4: Counterfactual Robustness ───
    print(f"\n{'=' * 60}")
    print("TEST 4: Counterfactual Robustness")
    print(f"{'=' * 60}")
    fact_samples = data.get("en_fact", [])[:n]

    if fact_samples:
        print(f"  Running {n} samples...", end=" ", flush=True)
        cf_result = run_counterfactual(harness, fact_samples)
        all_results["counterfactual"] = cf_result
        print("done")

    # ─── Summary ───
    print(f"\n{'=' * 60}")
    print("RGB BENCHMARK SUMMARY")
    print(f"{'=' * 60}")

    print("\n  Noise Robustness (gate pass rate):")
    print(f"    {'Noise':>8} {'Our Gate':>12} {'ChatGPT':>12} {'Target':>12}")
    chatgpt_noise = {0.0: 96.3, 0.2: 94.0, 0.4: 92.0, 0.6: 88.3}
    for ratio in NOISE_RATIOS:
        key = str(ratio)
        ours = noise_results.get(key, {}).get("gate_pass_rate", 0) * 100
        theirs = chatgpt_noise.get(ratio, 0)
        print(f"    {ratio:>8.1f} {ours:>11.1f}% {theirs:>11.1f}% {'>90':>12}")

    print("\n  Negative Rejection:")
    print(f"    Our rejection rate:  {neg_result['rejection_rate']:.1%}")
    print("    ChatGPT best (Rej*): 45.0%")
    print("    Target:              >60%")

    if int_results:
        print("\n  Information Integration:")
        for ratio, r in int_results.items():
            print(f"    Noise={ratio}: gate_pass={r['gate_pass_rate']:.1%}  (ChatGPT: 55%)")

    # Save
    out_path = os.path.join(SCRIPT_DIR, args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
