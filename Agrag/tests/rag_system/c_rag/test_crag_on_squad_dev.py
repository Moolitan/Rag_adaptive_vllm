"""
CRag Performance & QA Evaluation Script (FAISS + SQuAD)
Hop2Rag-style instrumentation with DataCollector
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
    get_llm_calls,
    get_prompt_token_distribution,
    get_performance_records,
    get_performance_summary,
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


def exact_match(pred: str, gt: str) -> int:
    return int(normalize_answer(pred) == normalize_answer(gt))


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

    em: int
    f1: float

    total_latency_sec: float

    total_nodes: int
    llm_nodes: int
    retriever_nodes: int
    cpu_nodes: int

    total_llm_latency_sec: float
    total_retriever_latency_sec: float
    total_input_tokens: int
    total_output_tokens: int

    used_web_search: bool

    records: List[Dict[str, Any]]
    llm_call_details: List[Dict[str, Any]]  # 每个 LLM call 的 prompt 和 response

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

    # 获取节点级别的执行记录
    records = get_performance_records()
    # 获取 LLM call 的详细记录（包含 prompt 和 response）
    llm_call_details = get_llm_calls()
    # 获取性能摘要
    summary = get_performance_summary()

    # ---- Hop2Rag-style classification ----
    llm_nodes = [r for r in records if r.get("node_type") == "llm"]
    retriever_nodes = [r for r in records if r.get("node_type") == "retriever"]
    cpu_nodes = [r for r in records if r.get("node_type") == "cpu"]

    total_llm_latency = summary.get("llm_latency_sec", 0)
    total_retriever_latency = summary.get("retriever_latency_sec", 0)

    # 从 llm_call_details 计算 token 统计
    total_input_tokens = sum(call.get("prompt_tokens", 0) for call in llm_call_details)
    total_output_tokens = sum(call.get("response_tokens", 0) for call in llm_call_details)

    pred = result.get("answer", "")

    em = max((exact_match(pred, a) for a in answers), default=0)
    f1 = max((f1_score(pred, a) for a in answers), default=0.0)

    metadata = result.get("metadata", {})

    return PerformanceResult(
        question=question,
        answer=pred,
        em=em,
        f1=f1,
        total_latency_sec=total_latency,

        total_nodes=len(records),
        llm_nodes=len(llm_nodes),
        retriever_nodes=len(retriever_nodes),
        cpu_nodes=len(cpu_nodes),

        total_llm_latency_sec=total_llm_latency,
        total_retriever_latency_sec=total_retriever_latency,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,

        used_web_search=bool(metadata.get("used_web_search", False)),
        records=list(records),
        llm_call_details=llm_call_details,
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
                f"LLM={perf.llm_nodes} | Ret={perf.retriever_nodes}"
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

        "llm_latency_mean": float(np.mean(
            [r.total_llm_latency_sec for r in results])),

        "retriever_latency_mean": float(np.mean(
            [r.total_retriever_latency_sec for r in results])),

        "llm_nodes_mean": float(np.mean([r.llm_nodes for r in results])),
        "retriever_nodes_mean": float(np.mean([r.retriever_nodes for r in results])),

        "input_tokens_mean": float(np.mean([r.total_input_tokens for r in results])),
        "output_tokens_mean": float(np.mean([r.total_output_tokens for r in results])),

        "web_search_rate": float(np.mean([r.used_web_search for r in results])),
    }

# =============================
# Main
# =============================

def main():
    ap = argparse.ArgumentParser("CRag Performance Test (Hop2Rag-style)")
    ap.add_argument("--squad-dev", type=str)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--monitor-interval", type=float, default=0.5)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not FAISS_DIR:
        print("[ERROR] AGRAG_FAISS_DIR not set")
        sys.exit(1)

    if args.squad_dev:
        samples = load_squad_dev(Path(args.squad_dev))
        samples = samples[args.start: args.start + args.limit]
    else:
        samples = [{
            "question": "Which city is the capital of China?",
            "answers": ["Beijing"]
        }]

    monitor = VLLMMonitor(
        url="http://localhost:6000/metrics",
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

    for k, v in stats.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "performance_results.json", "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    with open(RESULTS_DIR / "performance_stats.json", "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
