"""
Hop2Rag Performance Test - Concurrent Version

使用 ThreadPoolExecutor 实现并发测试，每个线程独立的 DataCollector
"""

import argparse
import sys
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from runner.VLLMMonitor import VLLMMonitor
from runner.LanggraphMonitor import DataCollector

# 导入 Hop2Rag 相关函数（但不使用全局 monitor）
from Rag.hop2_rag_performancy_concurrent import run_hop2_rag_with_collector

PERSIST_DIR = os.environ.get("AGRAG_PERSIST_DIR", "")
COLLECTION_NAME = os.environ.get("AGRAG_COLLECTION_NAME", "hotpot_fullwiki")

RESULTS_DIR = ROOT / "tests" / "results" / "hop2rag_performance_concurrent"


@dataclass
class PerformanceResult:
    """单次请求的性能结果"""
    worker_id: int
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
    # 时间戳
    start_timestamp: float
    end_timestamp: float
    # 详细记录
    records: List[Dict[str, Any]]
    llm_calls: List[Dict[str, Any]]


def run_single_test_concurrent(
    question: str,
    k: int,
    max_hops: int,
    worker_id: int
) -> PerformanceResult:
    """
    并发环境下运行单个问题测试

    每个线程使用独立的 DataCollector，避免线程安全问题
    """
    # 创建线程独立的 collector
    collector = DataCollector(
        debug=False,
        track_prompts=True,
        encoding_name="cl100k_base"
    )

    start_time = time.time()
    start_timestamp = start_time

    # 运行 Hop2Rag（传入独立的 collector）
    result = run_hop2_rag_with_collector(
        question=question,
        persist_dir=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
        k=k,
        max_hops=max_hops,
        collector=collector
    )

    end_time = time.time()
    total_latency = end_time - start_time

    # 从 collector 获取记录
    records = collector.get_records()
    llm_calls = collector.get_llm_calls()

    # 按节点类型分类
    llm_nodes = [r for r in records if r.get("node_type") == "llm"]
    retriever_nodes = [r for r in records if r.get("node_type") == "retriever"]
    cpu_nodes = [r for r in records if r.get("node_type") == "cpu"]

    # 计算延迟
    total_llm_latency = sum(r.get("llm_latency", 0) for r in llm_nodes)
    total_retriever_latency = sum(r.get("retriever_latency", 0) for r in retriever_nodes)

    return PerformanceResult(
        worker_id=worker_id,
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
        start_timestamp=start_timestamp,
        end_timestamp=end_time,
        records=list(records),
        llm_calls=list(llm_calls)
    )


def run_benchmark_concurrent(
    questions: List[str],
    k: int,
    max_hops: int,
    max_workers: int = 4,
    verbose: bool = True
) -> tuple[List[PerformanceResult], List[Dict[str, Any]]]:
    """
    并发运行批量测试

    Args:
        questions: 问题列表
        k: 检索 K 值
        max_hops: 最大跳数
        max_workers: 最大并发线程数
        verbose: 是否打印详细信息

    Returns:
        (results, all_llm_calls): 性能结果列表和所有 LLM call 记录
    """
    results = []
    all_llm_calls = []
    completed_count = 0
    total_count = len(questions)
    lock = Lock()

    print(f"\n启动并发测试 (max_workers={max_workers})")
    print(f"   总问题数: {total_count}")
    print()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_info = {
            executor.submit(run_single_test_concurrent, q, k, max_hops, i): (i, q)
            for i, q in enumerate(questions)
        }

        # 按完成顺序收集结果
        for future in as_completed(future_to_info):
            worker_id, question = future_to_info[future]

            with lock:
                completed_count += 1
                current_count = completed_count

            try:
                result = future.result()

                with lock:
                    results.append(result)
                    all_llm_calls.extend(result.llm_calls)

                if verbose:
                    print(f"[{current_count}/{total_count}] Worker {worker_id}: {question[:50]}...")
                    print(f"  Latency: {result.total_latency_sec:.2f}s, "
                          f"Nodes: {result.total_nodes} "
                          f"(LLM:{result.llm_nodes}, Ret:{result.retriever_nodes}, CPU:{result.cpu_nodes}), "
                          f"LLM time: {result.total_llm_latency_sec:.2f}s, "
                          f"Hops: {result.num_hops}")

            except Exception as e:
                print(f"[{current_count}/{total_count}] Worker {worker_id}: Error: {e}")
                import traceback
                traceback.print_exc()

    # 按时间戳排序（恢复时间顺序）
    results.sort(key=lambda x: x.start_timestamp)
    all_llm_calls.sort(key=lambda x: x.get("timestamp", 0))

    return results, all_llm_calls


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

    # 计算时间跨度（用于计算吞吐量）
    start_times = [r.start_timestamp for r in results]
    end_times = [r.end_timestamp for r in results]
    total_duration = max(end_times) - min(start_times) if start_times and end_times else 0
    throughput = len(results) / total_duration if total_duration > 0 else 0

    return {
        "total_requests": len(results),
        "total_duration_sec": float(total_duration),
        "throughput_qps": float(throughput),
        "latency_mean": float(np.mean(latencies)),
        "latency_median": float(np.median(latencies)),
        "latency_std": float(np.std(latencies)),
        "latency_p50": float(np.percentile(latencies, 50)),
        "latency_p90": float(np.percentile(latencies, 90)),
        "latency_p95": float(np.percentile(latencies, 95)),
        "latency_p99": float(np.percentile(latencies, 99)),
        "latency_min": float(np.min(latencies)),
        "latency_max": float(np.max(latencies)),
        "llm_latency_mean": float(np.mean(llm_latencies)),
        "llm_latency_median": float(np.median(llm_latencies)),
        "llm_latency_total": float(np.sum(llm_latencies)),
        "retriever_latency_mean": float(np.mean(retriever_latencies)),
        "retriever_latency_median": float(np.median(retriever_latencies)),
        "retriever_latency_total": float(np.sum(retriever_latencies)),
        "total_nodes_mean": float(np.mean(total_nodes)),
        "llm_nodes_mean": float(np.mean(llm_nodes)),
        "llm_nodes_total": int(np.sum(llm_nodes)),
        "retriever_nodes_mean": float(np.mean(retriever_nodes)),
        "retriever_nodes_total": int(np.sum(retriever_nodes)),
        "cpu_nodes_mean": float(np.mean(cpu_nodes)),
        "cpu_nodes_total": int(np.sum(cpu_nodes)),
        "hops_mean": float(np.mean(hops)),
        "hops_median": float(np.median(hops)),
        "hops_min": int(np.min(hops)),
        "hops_max": int(np.max(hops)),
    }


def save_results(
    results: List[PerformanceResult],
    stats: Dict[str, Any],
    llm_calls: List[Dict[str, Any]],
    output_dir: Path
):
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

    # 保存 LLM calls 和 prompt token 分布
    llm_calls_file = output_dir / "llm_calls.json"
    token_counts = [call.get("prompt_tokens", 0) for call in llm_calls if call.get("prompt_tokens", 0) > 0]

    llm_data = {
        "llm_calls": llm_calls,
        "distribution": {
            "total_calls": len(llm_calls),
            "total_tokens": sum(token_counts),
            "min_tokens": min(token_counts) if token_counts else 0,
            "max_tokens": max(token_counts) if token_counts else 0,
            "mean_tokens": sum(token_counts) / len(token_counts) if token_counts else 0,
            "median_tokens": sorted(token_counts)[len(token_counts) // 2] if token_counts else 0,
            "token_counts": token_counts,
        }
    }

    with open(llm_calls_file, 'w', encoding='utf-8') as f:
        json.dump(llm_data, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print(f"  - performance_results.json: {len(results)} requests")
    print(f"  - performance_stats.json: aggregated statistics")
    print(f"  - llm_calls.json: {len(llm_calls)} LLM calls")
    if token_counts:
        print(f"    Token distribution: min={min(token_counts)}, max={max(token_counts)}, mean={sum(token_counts)/len(token_counts):.1f}")


def print_summary(stats: Dict[str, Any]):
    """打印性能摘要"""
    print("\n" + "=" * 70)
    print("Performance Summary")
    print("=" * 70)

    print("\n[Overall]")
    print(f"  Total Requests: {stats.get('total_requests', 0)}")
    print(f"  Total Duration: {stats.get('total_duration_sec', 0):.2f}s")
    print(f"  Throughput: {stats.get('throughput_qps', 0):.2f} QPS")

    print("\n[Latency (seconds)]")
    print(f"  Mean:   {stats.get('latency_mean', 0):.2f}s")
    print(f"  Median: {stats.get('latency_median', 0):.2f}s")
    print(f"  Std:    {stats.get('latency_std', 0):.2f}s")
    print(f"  P50:    {stats.get('latency_p50', 0):.2f}s")
    print(f"  P90:    {stats.get('latency_p90', 0):.2f}s")
    print(f"  P95:    {stats.get('latency_p95', 0):.2f}s")
    print(f"  P99:    {stats.get('latency_p99', 0):.2f}s")
    print(f"  Min:    {stats.get('latency_min', 0):.2f}s")
    print(f"  Max:    {stats.get('latency_max', 0):.2f}s")

    print("\n[LLM Latency]")
    print(f"  Mean:   {stats.get('llm_latency_mean', 0):.2f}s")
    print(f"  Median: {stats.get('llm_latency_median', 0):.2f}s")
    print(f"  Total:  {stats.get('llm_latency_total', 0):.2f}s")

    print("\n[Retriever Latency]")
    print(f"  Mean:   {stats.get('retriever_latency_mean', 0):.2f}s")
    print(f"  Median: {stats.get('retriever_latency_median', 0):.2f}s")
    print(f"  Total:  {stats.get('retriever_latency_total', 0):.2f}s")

    print("\n[Nodes]")
    print(f"  Total Nodes (mean):     {stats.get('total_nodes_mean', 0):.1f}")
    print(f"  LLM Nodes (mean):       {stats.get('llm_nodes_mean', 0):.1f} (total: {stats.get('llm_nodes_total', 0)})")
    print(f"  Retriever Nodes (mean): {stats.get('retriever_nodes_mean', 0):.1f} (total: {stats.get('retriever_nodes_total', 0)})")
    print(f"  CPU Nodes (mean):       {stats.get('cpu_nodes_mean', 0):.1f} (total: {stats.get('cpu_nodes_total', 0)})")

    print("\n[Hops]")
    print(f"  Mean:   {stats.get('hops_mean', 0):.2f}")
    print(f"  Median: {stats.get('hops_median', 0):.2f}")
    print(f"  Min:    {stats.get('hops_min', 0)}")
    print(f"  Max:    {stats.get('hops_max', 0)}")



def main():
    ap = argparse.ArgumentParser(
        description="Hop2Rag Performance Test - Concurrent Version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法（4 个并发线程）
  python test_hop2rag_performance_concurrent.py \\
      --limit 20 \\
      --max-workers 4

  # 高并发压力测试
  python test_hop2rag_performance_concurrent.py \\
      --limit 100 \\
      --max-workers 16 \\
      --monitor-interval 0.05

  # 低并发功能测试
  python test_hop2rag_performance_concurrent.py \\
      --limit 10 \\
      --max-workers 2 \\
      --verbose
        """
    )

    ap.add_argument("--limit", type=int, default=10, help="Number of questions")
    ap.add_argument("--k", type=int, default=10, help="Retrieval K")
    ap.add_argument("--max-hops", type=int, default=3, help="Max hops")
    ap.add_argument("--max-workers", type=int, default=4, help="Max concurrent workers (threads)")
    ap.add_argument("--monitor-interval", type=float, default=0.5, help="VLLMMonitor polling interval")
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
        "What screenwriter with credits for \"Evolution\" co-wrote a film starring Nicolas Cage and Téa Leoni?",
        "What year did Guns N Roses perform a promo for a movie starring Arnold Schwarzenegger as a former New York Police detective?",
        "Are Random House Tower and 888 7th Avenue both used for real estate?",
        "The football manager who recruited David Beckham managed Manchester United during what timeframe?",
        "Brown State Fishing Lake is in a country that has a population of how many inhabitants?",
    ]
    questions = questions[:min(args.limit, len(questions))]

    print("=" * 70)
    print("Hop2Rag Performance Test - Concurrent Version")
    print("=" * 70)
    print(f"Questions: {len(questions)}")
    print(f"K: {args.k}, Max Hops: {args.max_hops}")
    print(f"Max Workers: {args.max_workers}")
    print(f"Monitor Interval: {args.monitor_interval}s")
    print()

    # 启动 VLLMMonitor
    monitor = VLLMMonitor(
        url="http://localhost:8000/metrics",
        interval=args.monitor_interval,
        csv_path=RESULTS_DIR / "vllm_metrics_concurrent.csv",
        flush_every=1
    )
    monitor.start()

    try:
        # 运行并发测试
        results, all_llm_calls = run_benchmark_concurrent(
            questions,
            args.k,
            args.max_hops,
            args.max_workers,
            args.verbose
        )
        stats = compute_stats(results)

    finally:
        monitor.stop()

    # 打印性能摘要
    print_summary(stats)

    # 保存结果
    save_results(results, stats, all_llm_calls, RESULTS_DIR)

    print("\n" + "=" * 70)
    print("Next Steps")
    print("=" * 70)
    print("\n1. 分析 prompt token 分布:")
    print(f"   python Agrag/tests/rag_system/Hop2rag/analyze_prompt_distribution.py \\")
    print(f"       --input {RESULTS_DIR}/llm_calls.json \\")
    print(f"       --output {RESULTS_DIR}/plots/prompt_dist \\")
    print(f"       --suggest-threshold")
    print("\n2. 绘制 prefix cache 命中率图:")
    print(f"   python Agrag/tests/rag_system/Hop2rag/plot_prefix_cache_hitrate.py \\")
    print(f"       --input {RESULTS_DIR}/vllm_metrics.csv \\")
    print(f"       --output {RESULTS_DIR}/plots/prefix_cache_hitrate.png")


if __name__ == "__main__":
    main()
