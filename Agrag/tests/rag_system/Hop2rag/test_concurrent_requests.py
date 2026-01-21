#!/usr/bin/env python3
"""
实验六：多请求并发资源竞争分析

目标：分析并发请求时的 GPU 内存和 vLLM batch 行为。

观察点：
- 不同并发度下的吞吐量
- 并发时的延迟分布（P50, P95, P99）
- GPU 内存峰值随并发度变化
- vLLM batch 效率
- 请求间的资源竞争和排队情况

输出：
- results/hop2rag/concurrency/concurrency_analysis.json
- results/hop2rag/concurrency/throughput_vs_concurrency.png
- results/hop2rag/concurrency/latency_vs_concurrency.png
- results/hop2rag/concurrency/memory_vs_concurrency.png
"""

import argparse
import json
import sys
import os
import time
import threading
import concurrent.futures
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import queue

# Setup paths
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# Check for CUDA
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARN] matplotlib not found, visualization disabled")

# Environment variables
PERSIST_DIR = os.environ.get("AGRAG_PERSIST_DIR", "")
COLLECTION_NAME = os.environ.get("AGRAG_COLLECTION_NAME", "hotpot_fullwiki")

# Results directory
RESULTS_DIR = ROOT / "tests" / "results" / "hop2rag" / "concurrency"


@dataclass
class RequestResult:
    """单请求结果"""
    request_id: str
    question: str
    concurrency_level: int

    # 时间
    submit_time: float
    start_time: float
    end_time: float

    # 延迟
    queue_time_ms: float  # 排队时间
    execution_time_ms: float  # 执行时间
    total_time_ms: float  # 总时间

    # 结果
    success: bool
    error: str = ""
    num_hops: int = 0


@dataclass
class ConcurrencyTestResult:
    """并发测试结果"""
    concurrency_level: int
    num_requests: int
    total_duration_sec: float

    # 吞吐量
    throughput_qps: float

    # 延迟统计
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_min_ms: float
    latency_max_ms: float

    # 排队时间
    queue_time_mean_ms: float
    queue_time_max_ms: float

    # 内存
    memory_peak_mb: float

    # 成功率
    success_rate: float

    # 原始数据
    requests: List[RequestResult] = field(default_factory=list)


class ConcurrencyTester:
    """并发测试器"""

    def __init__(self, persist_dir: str, collection_name: str, k: int, max_hops: int):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.k = k
        self.max_hops = max_hops

        # 共享状态
        self.active_requests = 0
        self.lock = threading.Lock()
        self.memory_samples = []
        self.memory_sampler_running = False

    def _sample_memory(self):
        """内存采样线程"""
        while self.memory_sampler_running:
            if HAS_CUDA:
                mem = torch.cuda.memory_allocated() / 1024**2
                self.memory_samples.append((time.time(), mem))
            time.sleep(0.1)

    def run_single_request(
        self,
        request_id: str,
        question: str,
        concurrency_level: int,
        submit_time: float
    ) -> RequestResult:
        """执行单个请求"""
        from Rag.hop2_rag import run_hop2_rag

        with self.lock:
            self.active_requests += 1

        start_time = time.time()
        queue_time = (start_time - submit_time) * 1000

        result = RequestResult(
            request_id=request_id,
            question=question,
            concurrency_level=concurrency_level,
            submit_time=submit_time,
            start_time=start_time,
            end_time=0,
            queue_time_ms=queue_time,
            execution_time_ms=0,
            total_time_ms=0,
            success=False,
        )

        try:
            rag_result = run_hop2_rag(
                question=question,
                persist_dir=self.persist_dir,
                collection_name=self.collection_name,
                k=self.k,
                max_hops=self.max_hops
            )

            result.success = True
            result.num_hops = rag_result.get("metadata", {}).get("total_hops", 0)

        except Exception as e:
            result.success = False
            result.error = str(e)

        finally:
            end_time = time.time()
            result.end_time = end_time
            result.execution_time_ms = (end_time - start_time) * 1000
            result.total_time_ms = (end_time - submit_time) * 1000

            with self.lock:
                self.active_requests -= 1

        return result

    def run_concurrent_test(
        self,
        questions: List[str],
        concurrency: int,
        verbose: bool = True
    ) -> ConcurrencyTestResult:
        """运行并发测试"""

        if verbose:
            print(f"\n  Testing concurrency={concurrency} with {len(questions)} requests...")

        # 重置内存采样
        self.memory_samples = []
        if HAS_CUDA:
            torch.cuda.reset_peak_memory_stats()

        # 启动内存采样
        self.memory_sampler_running = True
        memory_thread = threading.Thread(target=self._sample_memory, daemon=True)
        memory_thread.start()

        results: List[RequestResult] = []
        submit_time = time.time()

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = []

                for i, question in enumerate(questions):
                    request_id = f"req_{i:04d}_c{concurrency}"
                    future = executor.submit(
                        self.run_single_request,
                        request_id,
                        question,
                        concurrency,
                        time.time()  # 每个请求的提交时间
                    )
                    futures.append(future)

                # 收集结果
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    result = future.result()
                    results.append(result)

                    if verbose and (i + 1) % 5 == 0:
                        print(f"    Completed {i+1}/{len(questions)} requests")

        finally:
            self.memory_sampler_running = False
            memory_thread.join(timeout=1.0)

        total_duration = time.time() - submit_time

        # 计算统计
        successful = [r for r in results if r.success]
        latencies = [r.total_time_ms for r in successful]
        queue_times = [r.queue_time_ms for r in results]

        if not latencies:
            latencies = [0]

        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)

        # 内存峰值
        memory_peak = 0
        if HAS_CUDA:
            memory_peak = torch.cuda.max_memory_allocated() / 1024**2
        elif self.memory_samples:
            memory_peak = max(m for _, m in self.memory_samples)

        return ConcurrencyTestResult(
            concurrency_level=concurrency,
            num_requests=len(results),
            total_duration_sec=total_duration,

            throughput_qps=len(results) / total_duration if total_duration > 0 else 0,

            latency_mean_ms=sum(latencies) / n,
            latency_p50_ms=latencies_sorted[int(n * 0.50)] if n > 0 else 0,
            latency_p95_ms=latencies_sorted[min(int(n * 0.95), n-1)] if n > 0 else 0,
            latency_p99_ms=latencies_sorted[min(int(n * 0.99), n-1)] if n > 0 else 0,
            latency_min_ms=min(latencies) if latencies else 0,
            latency_max_ms=max(latencies) if latencies else 0,

            queue_time_mean_ms=sum(queue_times) / len(queue_times) if queue_times else 0,
            queue_time_max_ms=max(queue_times) if queue_times else 0,

            memory_peak_mb=memory_peak,
            success_rate=len(successful) / len(results) if results else 0,

            requests=results,
        )


def plot_throughput_vs_concurrency(results: List[ConcurrencyTestResult], output_path: str):
    """绘制吞吐量 vs 并发度"""
    if not HAS_MATPLOTLIB or not results:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    concurrencies = [r.concurrency_level for r in results]
    throughputs = [r.throughput_qps for r in results]

    ax.plot(concurrencies, throughputs, 'o-', linewidth=2, markersize=10, color='steelblue')

    # 标注数值
    for c, t in zip(concurrencies, throughputs):
        ax.annotate(f'{t:.2f}', xy=(c, t), xytext=(5, 5),
                   textcoords='offset points', fontsize=10)

    ax.set_xlabel('Concurrency Level', fontsize=12)
    ax.set_ylabel('Throughput (QPS)', fontsize=12)
    ax.set_title('Throughput vs Concurrency\n(Higher is better)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(concurrencies)

    # 标注理想线性扩展
    if len(concurrencies) > 1:
        base_throughput = throughputs[0]
        ideal = [base_throughput * c / concurrencies[0] for c in concurrencies]
        ax.plot(concurrencies, ideal, '--', color='gray', alpha=0.5, label='Ideal Linear Scaling')
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved throughput plot: {output_path}")


def plot_latency_vs_concurrency(results: List[ConcurrencyTestResult], output_path: str):
    """绘制延迟 vs 并发度"""
    if not HAS_MATPLOTLIB or not results:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    concurrencies = [r.concurrency_level for r in results]
    p50s = [r.latency_p50_ms for r in results]
    p95s = [r.latency_p95_ms for r in results]
    p99s = [r.latency_p99_ms for r in results]

    x = np.arange(len(concurrencies))
    width = 0.25

    ax.bar(x - width, p50s, width, label='P50', color='steelblue', alpha=0.8)
    ax.bar(x, p95s, width, label='P95', color='coral', alpha=0.8)
    ax.bar(x + width, p99s, width, label='P99', color='forestgreen', alpha=0.8)

    ax.set_xlabel('Concurrency Level', fontsize=12)
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('Latency Percentiles vs Concurrency\n(Lower is better)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(concurrencies)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved latency plot: {output_path}")


def plot_memory_vs_concurrency(results: List[ConcurrencyTestResult], output_path: str):
    """绘制内存 vs 并发度"""
    if not HAS_MATPLOTLIB or not results:
        return

    memories = [r.memory_peak_mb for r in results]
    if all(m == 0 for m in memories):
        print("[WARN] No memory data available")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    concurrencies = [r.concurrency_level for r in results]

    ax.bar(concurrencies, memories, color='coral', alpha=0.8, edgecolor='black')

    for c, m in zip(concurrencies, memories):
        ax.text(c, m + 50, f'{m:.0f}', ha='center', fontsize=10)

    ax.set_xlabel('Concurrency Level', fontsize=12)
    ax.set_ylabel('Peak Memory (MB)', fontsize=12)
    ax.set_title('GPU Memory Peak vs Concurrency', fontsize=14)
    ax.set_xticks(concurrencies)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved memory plot: {output_path}")


def plot_latency_distribution(results: List[ConcurrencyTestResult], output_path: str):
    """绘制延迟分布箱线图"""
    if not HAS_MATPLOTLIB or not results:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    data = []
    labels = []

    for r in results:
        latencies = [req.total_time_ms for req in r.requests if req.success]
        if latencies:
            data.append(latencies)
            labels.append(f'C={r.concurrency_level}')

    if not data:
        return

    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(data)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel('Concurrency Level', fontsize=12)
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('Latency Distribution by Concurrency Level', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved latency distribution: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Concurrent Requests Analysis for Hop2Rag"
    )
    parser.add_argument("--num-requests", type=int, default=20, help="Total requests per concurrency level")
    parser.add_argument("--concurrency", type=str, default="1,2,4", help="Concurrency levels (comma-separated)")
    parser.add_argument("--k", type=int, default=10, help="Documents per hop")
    parser.add_argument("--max-hops", type=int, default=3, help="Maximum hops")
    parser.add_argument("--skip-plots", action="store_true", help="Skip visualization")

    args = parser.parse_args()

    # 解析并发度列表
    concurrency_levels = [int(c.strip()) for c in args.concurrency.split(",")]

    # 检查环境变量
    if not PERSIST_DIR:
        print("[ERROR] Please set AGRAG_PERSIST_DIR environment variable")
        sys.exit(1)

    print("=" * 70)
    print("Concurrent Requests Analysis")
    print("=" * 70)
    print(f"Requests per level: {args.num_requests}")
    print(f"Concurrency levels: {concurrency_levels}")
    print(f"K:                  {args.k}")
    print(f"Max Hops:           {args.max_hops}")
    print(f"CUDA:               {HAS_CUDA}")
    print(f"PERSIST_DIR:        {PERSIST_DIR}")
    print("=" * 70)

    # 创建输出目录
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 示例问题
    sample_questions = [
        "Who is the director of the movie that won Best Picture at the 2020 Oscars?",
        "What is the capital of the country where the Eiffel Tower is located?",
        "Which university did the founder of Microsoft graduate from?",
        "What is the population of the city where the 2008 Olympics were held?",
        "Who wrote the book that inspired the movie The Shawshank Redemption?",
        "What is the currency used in the country that hosted the 2014 FIFA World Cup?",
        "Who is the CEO of the company that created the iPhone?",
        "What language is spoken in the country where Mount Fuji is located?",
        "Who directed the movie based on the novel by J.R.R. Tolkien?",
        "What is the GDP of the country where Big Ben is located?",
    ]
    questions = (sample_questions * ((args.num_requests // len(sample_questions)) + 1))[:args.num_requests]

    # 创建测试器
    tester = ConcurrencyTester(
        persist_dir=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
        k=args.k,
        max_hops=args.max_hops
    )

    # 运行测试
    all_results: List[ConcurrencyTestResult] = []

    for concurrency in concurrency_levels:
        result = tester.run_concurrent_test(
            questions=questions,
            concurrency=concurrency,
            verbose=True
        )
        all_results.append(result)

        print(f"\n  Concurrency {concurrency} Results:")
        print(f"    Throughput:    {result.throughput_qps:.2f} QPS")
        print(f"    Latency P50:   {result.latency_p50_ms:.0f} ms")
        print(f"    Latency P95:   {result.latency_p95_ms:.0f} ms")
        print(f"    Memory Peak:   {result.memory_peak_mb:.0f} MB")
        print(f"    Success Rate:  {result.success_rate:.1%}")

    # 保存结果
    output_json = RESULTS_DIR / "concurrency_analysis.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump({
            "config": {
                "num_requests": args.num_requests,
                "concurrency_levels": concurrency_levels,
                "k": args.k,
                "max_hops": args.max_hops,
            },
            "results": [
                {
                    "concurrency_level": r.concurrency_level,
                    "num_requests": r.num_requests,
                    "total_duration_sec": r.total_duration_sec,
                    "throughput_qps": r.throughput_qps,
                    "latency_mean_ms": r.latency_mean_ms,
                    "latency_p50_ms": r.latency_p50_ms,
                    "latency_p95_ms": r.latency_p95_ms,
                    "latency_p99_ms": r.latency_p99_ms,
                    "queue_time_mean_ms": r.queue_time_mean_ms,
                    "memory_peak_mb": r.memory_peak_mb,
                    "success_rate": r.success_rate,
                }
                for r in all_results
            ]
        }, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Saved results: {output_json}")

    # 绘图
    if not args.skip_plots and HAS_MATPLOTLIB:
        print("\nGenerating visualizations...")
        plot_throughput_vs_concurrency(all_results, str(RESULTS_DIR / "throughput_vs_concurrency.png"))
        plot_latency_vs_concurrency(all_results, str(RESULTS_DIR / "latency_vs_concurrency.png"))
        plot_memory_vs_concurrency(all_results, str(RESULTS_DIR / "memory_vs_concurrency.png"))
        plot_latency_distribution(all_results, str(RESULTS_DIR / "latency_distribution.png"))

    # 打印总结
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Concurrency':<12} {'QPS':<10} {'P50(ms)':<10} {'P95(ms)':<10} {'Memory(MB)':<12} {'Success':<8}")
    print("-" * 70)

    for r in all_results:
        print(f"{r.concurrency_level:<12} {r.throughput_qps:<10.2f} {r.latency_p50_ms:<10.0f} "
              f"{r.latency_p95_ms:<10.0f} {r.memory_peak_mb:<12.0f} {r.success_rate:<8.1%}")

    print()

    # 分析
    if len(all_results) > 1:
        base = all_results[0]
        best_qps = max(all_results, key=lambda x: x.throughput_qps)

        print("Analysis:")
        print(f"  - Best throughput at concurrency={best_qps.concurrency_level}: {best_qps.throughput_qps:.2f} QPS")

        scaling_efficiency = best_qps.throughput_qps / (base.throughput_qps * best_qps.concurrency_level / base.concurrency_level)
        print(f"  - Scaling efficiency: {scaling_efficiency:.1%}")

        if scaling_efficiency < 0.5:
            print("  - Poor scaling: consider investigating bottlenecks")
            print("    * vLLM batching efficiency")
            print("    * GPU memory contention")
            print("    * CPU-bound preprocessing")
        elif scaling_efficiency < 0.8:
            print("  - Moderate scaling: some overhead present")
        else:
            print("  - Good scaling efficiency!")

    print()
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
