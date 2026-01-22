"""
Hop2Rag Performance Test with DataCollector

使用 runner.performancemonitor.DataCollector 监控 Hop2Rag 工作流的性能指标。
"""

import argparse
import sys
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from runner.VLLMMonitor import VLLMMonitor

from Rag.hop2_rag_performancy import (
    run_hop2_rag,
    get_performance_records,
    clear_performance_records,
)

PERSIST_DIR = os.environ.get("AGRAG_PERSIST_DIR", "")
COLLECTION_NAME = os.environ.get("AGRAG_COLLECTION_NAME", "hotpot_fullwiki")

RESULTS_DIR = ROOT / "tests" / "results" / "hop2rag_performance"


@dataclass
class PerformanceResult:
    """单次请求的性能结果"""
    question: str
    answer: str
    num_hops: int
    total_latency_sec: float
    llm_calls: int
    total_llm_latency_sec: float
    total_input_tokens: int
    total_output_tokens: int
    node_executions: int
    records: List[Dict[str, Any]]


def run_single_test(question: str, k: int, max_hops: int) -> PerformanceResult:
    """运行单个问题并收集性能数据"""
    clear_performance_records()

    start = time.time()
    result = run_hop2_rag(
        question=question,
        persist_dir=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
        k=k,
        max_hops=max_hops
    )
    total_latency = time.time() - start

    records = get_performance_records()

    llm_records = [r for r in records if r.get("event") == "llm_call"]
    node_records = [r for r in records if r.get("event") == "node_execution"]

    total_llm_latency = sum(r.get("latency", 0) for r in llm_records)
    total_input_tokens = sum(r.get("input_tokens", 0) for r in llm_records)
    total_output_tokens = sum(r.get("output_tokens", 0) for r in llm_records)

    return PerformanceResult(
        question=question,
        answer=result.get("answer", ""),
        num_hops=result.get("current_hop", 0),
        total_latency_sec=total_latency,
        llm_calls=len(llm_records),
        total_llm_latency_sec=total_llm_latency,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        node_executions=len(node_records),
        records=list(records)
    )


def run_benchmark(questions: List[str], k: int, max_hops: int, verbose: bool = True) -> List[PerformanceResult]:
    """运行批量测试"""
    results = []

    for i, q in enumerate(questions):
        if verbose:
            print(f"[{i+1}/{len(questions)}] {q[:60]}...")

        perf = run_single_test(q, k, max_hops)
        results.append(perf)

        if verbose:
            print(f"  Latency: {perf.total_latency_sec:.2f}s, "
                  f"LLM calls: {perf.llm_calls}, "
                  f"Tokens: {perf.total_input_tokens}+{perf.total_output_tokens}")

    return results


def compute_stats(results: List[PerformanceResult]) -> Dict[str, Any]:
    """计算统计数据"""
    if not results:
        return {}

    latencies = [r.total_latency_sec for r in results]
    llm_latencies = [r.total_llm_latency_sec for r in results]
    llm_calls = [r.llm_calls for r in results]
    input_tokens = [r.total_input_tokens for r in results]
    output_tokens = [r.total_output_tokens for r in results]
    hops = [r.num_hops for r in results]

    return {
        "total_requests": len(results),
        "latency_mean": float(np.mean(latencies)),
        "latency_p50": float(np.percentile(latencies, 50)),
        "latency_p95": float(np.percentile(latencies, 95)),
        "llm_latency_mean": float(np.mean(llm_latencies)),
        "llm_calls_mean": float(np.mean(llm_calls)),
        "input_tokens_mean": float(np.mean(input_tokens)),
        "output_tokens_mean": float(np.mean(output_tokens)),
        "hops_mean": float(np.mean(hops)),
    }


def save_results(results: List[PerformanceResult], stats: Dict[str, Any], output_dir: Path):
    """保存结果"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存详细结果
    results_file = output_dir / "performance_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)

    # 保存统计
    stats_file = output_dir / "performance_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_dir}")


def main():
    ap = argparse.ArgumentParser(description="Hop2Rag Performance Test with DataCollector")
    ap.add_argument("--limit", type=int, default=5, help="Number of questions")
    ap.add_argument("--k", type=int, default=10, help="Retrieval K")
    ap.add_argument("--max-hops", type=int, default=3, help="Max hops")
    ap.add_argument("--monitor-interval", type=float, default=0.5, help="Monitor polling interval")                                                                                   
    ap.add_argument("--verbose", action="store_true", help="Verbose output")
    args = ap.parse_args()

    if not PERSIST_DIR:
        print("[ERROR] Set AGRAG_PERSIST_DIR environment variable")
        sys.exit(1)

    # 示例问题
    questions = [
        # "Were Scott Derrickson and Ed Wood of the same nationality?",
        # "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
        "What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?",
        # "Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?",
        "The director of the romantic comedy \"Big Stone Gap\" is based in what New York city?",
        "2014 S/S is the debut album of a South Korean boy group that was formed by who?",
        # "Who was known by his stage name Aladin and helped organizations improve their performance as a consultant?",
        "The arena where the Lewiston Maineiacs played their home games can seat how many people?",
        # "Who is older, Annie Morton or Terry Richardson?",
        "Are Local H and For Against both from the United States?",
        # "What is the name of the fight song of the university whose main campus is in Lawrence, Kansas and whose branch campuses are in the Kansas City metropolitan area?",
        "What screenwriter with credits for \"Evolution\" co-wrote a film starring Nicolas Cage and Téa Leoni?",
        # "What year did Guns N Roses perform a promo for a movie starring Arnold Schwarzenegger as a former New York Police detective?",
        # "Are Random House Tower and 888 7th Avenue both used for real estate?",
        # "The football manager who recruited David Beckham managed Manchester United during what timeframe?",
        # "Brown State Fishing Lake is in a country that has a population of how many inhabitants ?",
    ]
    questions = questions[:min(args.limit, len(questions))]

    print("=" * 60)
    print("Hop2Rag Performance Test (DataCollector)")
    print("=" * 60)
    print(f"Questions: {len(questions)}")
    print(f"K: {args.k}, Max Hops: {args.max_hops}")
    print()

    monitor = VLLMMonitor(
            url="http://localhost:8000/metrics",
            interval=args.monitor_interval,
            csv_path=RESULTS_DIR / "vllm_metrics.csv",
            flush_every = 1)
    monitor.start() # 启动监控, 创建一个新的线程
    try:
        results = run_benchmark(questions, args.k, args.max_hops, args.verbose)
        stats = compute_stats(results)
    finally:
        monitor.stop()

    print("\n" + "=" * 60)
    print("Statistics")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

    save_results(results, stats, RESULTS_DIR)


if __name__ == "__main__":
    main()
