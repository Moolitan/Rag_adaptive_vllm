"""
CRag Performance & QA Evaluation Script (FAISS + SQuAD)

- Corrective RAG (CRag)
- Vector Store: FAISS (AGRAG_FAISS_DIR)
- QA Dataset: SQuAD dev.json
- Metrics:
    - Performance: latency / LLM calls / tokens / web search
    - Quality: EM / F1
- Monitoring: VLLMMonitor + DataCollector
"""

import argparse
import sys
import os
import json
import time
import re
import string
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from collections import Counter
import numpy as np

# =============================
# Path setup
# =============================

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# =============================
# Imports
# =============================

from runner.VLLMMonitor import VLLMMonitor

from Rag.c_rag_performancy import (
    run_c_rag,
    get_performance_records,
    clear_performance_records,
)

# =============================
# Environment
# =============================

FAISS_DIR = os.environ.get("AGRAG_FAISS_DIR", "")
RESULTS_DIR = ROOT / "tests" / "results" / "crag_performance"

# =============================
# SQuAD utilities
# =============================

def load_squad_dev(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for article in data["data"]:
        for para in article["paragraphs"]:
            for qa in para["qas"]:
                samples.append({
                    "id": qa["id"],
                    "question": qa["question"],
                    "answers": [a["text"] for a in qa["answers"]],
                })
    return samples


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(f"[{string.punctuation}]", "", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gt: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gt)


def f1_score(pred: str, gt: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gt_tokens = normalize_answer(gt).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)

# =============================
# Data structure
# =============================

@dataclass
class PerformanceResult:
    question: str
    answer: str

    em: bool
    f1: float

    total_latency_sec: float

    llm_calls: int
    total_llm_latency_sec: float
    total_input_tokens: int
    total_output_tokens: int

    node_executions: int
    used_web_search: bool
    web_search_rounds: int

    records: List[Dict[str, Any]]

# =============================
# Single test
# =============================

def run_single_test(sample: Dict[str, Any]) -> PerformanceResult:
    clear_performance_records()

    question = sample["question"]
    answers = sample.get("answers", [])

    start = time.time()
    result = run_c_rag(question)
    total_latency = time.time() - start

    records = get_performance_records()

    llm_records = [r for r in records if r.get("event") == "llm_call"]
    node_records = [r for r in records if r.get("event") == "node_execution"]

    pred = result.get("answer", "")

    em = max((exact_match(pred, a) for a in answers), default=False)
    f1 = max((f1_score(pred, a) for a in answers), default=0.0)

    metadata = result.get("metadata", {})

    return PerformanceResult(
        question=question,
        answer=pred,
        em=em,
        f1=f1,

        total_latency_sec=total_latency,

        llm_calls=len(llm_records),
        total_llm_latency_sec=sum(r.get("latency", 0) for r in llm_records),
        total_input_tokens=sum(r.get("input_tokens", 0) for r in llm_records),
        total_output_tokens=sum(r.get("output_tokens", 0) for r in llm_records),

        node_executions=len(node_records),
        used_web_search=bool(metadata.get("used_web_search", False)),
        web_search_rounds=int(metadata.get("web_search_rounds", 0)),

        records=list(records),
    )

# =============================
# Batch benchmark
# =============================

def run_benchmark(samples: List[Dict[str, Any]], verbose: bool):
    results = []
    for i, sample in enumerate(samples):
        if verbose:
            print(f"[{i+1}/{len(samples)}] {sample['question'][:80]}")

        perf = run_single_test(sample)
        results.append(perf)

        if verbose:
            print(
                f"  EM={perf.em} | F1={perf.f1:.3f} | "
                f"Latency={perf.total_latency_sec:.2f}s | "
                f"LLM={perf.llm_calls} | "
                f"Web={perf.used_web_search}"
            )

    return results

# =============================
# Statistics
# =============================

def compute_stats(results: List[PerformanceResult]) -> Dict[str, Any]:
    if not results:
        return {}

    return {
        "total_requests": len(results),
        "EM": float(np.mean([r.em for r in results])),
        "F1": float(np.mean([r.f1 for r in results])),

        "latency_mean": float(np.mean([r.total_latency_sec for r in results])),
        "latency_p95": float(np.percentile(
            [r.total_latency_sec for r in results], 95)),

        "llm_calls_mean": float(np.mean([r.llm_calls for r in results])),
        "input_tokens_mean": float(np.mean([r.total_input_tokens for r in results])),
        "output_tokens_mean": float(np.mean([r.total_output_tokens for r in results])),

        "web_search_rate": float(np.mean([r.used_web_search for r in results])),
    }

# =============================
# Save results
# =============================

def save_results(results, stats, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "performance_results.json", "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)

    with open(out_dir / "performance_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {out_dir}")

# =============================
# Main
# =============================

def main():
    ap = argparse.ArgumentParser("CRag Performance Test (FAISS + SQuAD)")
    ap.add_argument("--squad-dev", type=str, help="Path to SQuAD dev.json")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--monitor-interval", type=float, default=0.5)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not FAISS_DIR:
        print("[ERROR] AGRAG_FAISS_DIR not set")
        sys.exit(1)

    if args.squad_dev:
        all_samples = load_squad_dev(Path(args.squad_dev))
        samples = all_samples[args.start: args.start + args.limit]
    else:
        samples = [{
            "question": "Which city is the capital of China?",
            "answers": ["Beijing"]
        }]

    print("=" * 60)
    print("CRag Performance & QA Evaluation")
    print("=" * 60)
    print(f"Questions: {len(samples)}\n")

    monitor = VLLMMonitor(
        url="http://localhost:8000/metrics",
        interval=args.monitor_interval,
        csv_path=RESULTS_DIR / "vllm_metrics.csv",
        flush_every=1,
    )

    monitor.start()
    try:
        results = run_benchmark(samples, args.verbose)
        stats = compute_stats(results)
    finally:
        monitor.stop()

    print("\nStatistics")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    save_results(results, stats, RESULTS_DIR)


if __name__ == "__main__":
    main()
