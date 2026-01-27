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
    # 节点统计
    total_nodes: int
    llm_nodes: int
    retriever_nodes: int
    cpu_nodes: int
    # 延迟统计
    total_llm_latency_sec: float
    total_retriever_latency_sec: float
    # 详细记录
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

    # 按节点类型分类
    llm_nodes = [r for r in records if r.get("node_type") == "llm"]
    retriever_nodes = [r for r in records if r.get("node_type") == "retriever"]
    cpu_nodes = [r for r in records if r.get("node_type") == "cpu"]

    # 计算延迟
    total_llm_latency = sum(r.get("llm_latency", 0) for r in llm_nodes)
    total_retriever_latency = sum(r.get("retriever_latency", 0) for r in retriever_nodes)

    return PerformanceResult(
        question=question,
        answer=result.get("answer", ""),
        num_hops=result.get("current_hop", 0),
        total_latency_sec=total_latency,
        total_nodes=len(records),
        llm_nodes=len(llm_nodes),
        retriever_nodes=len(retriever_nodes),
        cpu_nodes=len(cpu_nodes),
        total_llm_latency_sec=total_llm_latency,
        total_retriever_latency_sec=total_retriever_latency,
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
                  f"Nodes: {perf.total_nodes} (LLM:{perf.llm_nodes}, Ret:{perf.retriever_nodes}, CPU:{perf.cpu_nodes}), "
                  f"LLM time: {perf.total_llm_latency_sec:.2f}s")

    return results


def compute_stats(results: List[PerformanceResult]) -> Dict[str, Any]:
    """计算统计数据"""
    if not results:
        return {}

    latencies = [r.total_latency_sec for r in results]
    llm_latencies = [r.total_llm_latency_sec for r in results]
    retriever_latencies = [r.total_retriever_latency_sec for r in results]
    llm_nodes = [r.llm_nodes for r in results]
    retriever_nodes = [r.retriever_nodes for r in results]
    cpu_nodes = [r.cpu_nodes for r in results]
    total_nodes = [r.total_nodes for r in results]
    hops = [r.num_hops for r in results]

    return {
        "total_requests": len(results),
        "latency_mean": float(np.mean(latencies)),
        "latency_p50": float(np.percentile(latencies, 50)),
        "latency_p95": float(np.percentile(latencies, 95)),
        "llm_latency_mean": float(np.mean(llm_latencies)),
        "retriever_latency_mean": float(np.mean(retriever_latencies)),
        "total_nodes_mean": float(np.mean(total_nodes)),
        "llm_nodes_mean": float(np.mean(llm_nodes)),
        "retriever_nodes_mean": float(np.mean(retriever_nodes)),
        "cpu_nodes_mean": float(np.mean(cpu_nodes)),
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
        "Were Scott Derrickson and Ed Wood of the same nationality?",
        "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
        "What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?",
        "Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?",
        "The director of the romantic comedy \"Big Stone Gap\" is based in what New York city?",
        "2014 S/S is the debut album of a South Korean boy group that was formed by who?",
        "Who was known by his stage name Aladin and helped organizations improve their performance as a consultant?",
        "The arena where the Lewiston Maineiacs played their home games can seat how many people?",
        "Who is older, Annie Morton or Terry Richardson?",
        "Are Local H and For Against both from the United States?",
        "What is the name of the fight song of the university whose main campus is in Lawrence, Kansas and whose branch campuses are in the Kansas City metropolitan area?",
        # "What screenwriter with credits for \"Evolution\" co-wrote a film starring Nicolas Cage and Téa Leoni?",
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
