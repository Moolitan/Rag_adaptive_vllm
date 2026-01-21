#!/usr/bin/env python3
"""
实验四：vLLM 服务端指标观测

目标：观察 vLLM 内部的 KV Cache 命中、batch 行为、preemption 等指标。

观察点：
- KV Cache 命中率（需要启用 prefix caching）
- GPU Cache 使用率
- 请求吞吐量
- Batch 大小分布
- Preemption（抢占）情况

前置条件：
  启动 vLLM 时启用 prefix caching:
  python -m vllm.entrypoints.openai.api_server \
      --model Qwen2.5 \
      --enable-prefix-caching \
      --disable-log-requests

输出：
- results/hop2rag/vllm_metrics/metrics_timeline.json
- results/hop2rag/vllm_metrics/cache_hit_rate.png
- results/hop2rag/vllm_metrics/throughput.png
"""

import argparse
import json
import sys
import os
import time
import threading
import re
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional
from collections import defaultdict

import requests

# Setup paths
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

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
RESULTS_DIR = ROOT / "tests" / "results" / "hop2rag" / "vllm_metrics"


@dataclass
class VLLMMetricsSample:
    """vLLM 指标采样点"""
    timestamp: float
    relative_time_ms: float

    # Cache metrics
    gpu_cache_usage_perc: float = 0.0
    cpu_cache_usage_perc: float = 0.0
    prefix_cache_hit_rate: float = 0.0
    prefix_cache_queries: int = 0

    # Request metrics
    num_requests_running: int = 0
    num_requests_waiting: int = 0
    num_requests_swapped: int = 0

    # Throughput metrics
    prompt_throughput_toks_per_s: float = 0.0
    generation_throughput_toks_per_s: float = 0.0

    # Preemption
    num_preemptions: int = 0

    # Batch info
    num_batched_tokens: int = 0


class VLLMMetricsCollector:
    """vLLM 指标采集器"""

    def __init__(self, vllm_endpoint: str = "http://localhost:8000", sample_interval_ms: int = 100):
        self.endpoint = vllm_endpoint.rstrip('/')
        self.metrics_url = f"{self.endpoint}/metrics"
        self.sample_interval = sample_interval_ms / 1000
        self.samples: List[VLLMMetricsSample] = []
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.start_time: float = 0
        self.lock = threading.Lock()

        # 检查连接
        self._check_connection()

    def _check_connection(self):
        """检查 vLLM 连接"""
        try:
            resp = requests.get(self.metrics_url, timeout=5)
            if resp.status_code == 200:
                print(f"[INFO] Connected to vLLM at {self.endpoint}")

                # 检查是否启用了 prefix caching
                if "prefix_cache" in resp.text:
                    print("[INFO] Prefix caching metrics detected")
                else:
                    print("[WARN] Prefix caching metrics not found. Consider starting vLLM with --enable-prefix-caching")
            else:
                print(f"[WARN] vLLM metrics endpoint returned {resp.status_code}")
        except Exception as e:
            print(f"[WARN] Cannot connect to vLLM at {self.endpoint}: {e}")

    def _parse_prometheus_metrics(self, text: str) -> Dict[str, float]:
        """解析 Prometheus 格式的指标"""
        metrics = {}

        for line in text.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # 解析 metric_name{labels} value 格式
            match = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)\{?[^}]*\}?\s+([0-9.eE+-]+)$', line)
            if match:
                name = match.group(1)
                try:
                    value = float(match.group(2))
                    # 保留最新值（有些指标可能有多个 label 组合）
                    metrics[name] = value
                except ValueError:
                    pass

        return metrics

    def _fetch_metrics(self) -> Optional[VLLMMetricsSample]:
        """获取一次指标"""
        try:
            resp = requests.get(self.metrics_url, timeout=2)
            if resp.status_code != 200:
                return None

            metrics = self._parse_prometheus_metrics(resp.text)
            now = time.time()

            return VLLMMetricsSample(
                timestamp=now,
                relative_time_ms=(now - self.start_time) * 1000,

                # Cache metrics
                gpu_cache_usage_perc=metrics.get('vllm:gpu_cache_usage_perc', 0),
                cpu_cache_usage_perc=metrics.get('vllm:cpu_cache_usage_perc', 0),
                prefix_cache_hit_rate=metrics.get('vllm:prefix_cache_hit_rate', 0),
                prefix_cache_queries=int(metrics.get('vllm:prefix_cache_queries_total', 0)),

                # Request metrics
                num_requests_running=int(metrics.get('vllm:num_requests_running', 0)),
                num_requests_waiting=int(metrics.get('vllm:num_requests_waiting', 0)),
                num_requests_swapped=int(metrics.get('vllm:num_requests_swapped', 0)),

                # Throughput
                prompt_throughput_toks_per_s=metrics.get('vllm:avg_prompt_throughput_toks_per_s', 0),
                generation_throughput_toks_per_s=metrics.get('vllm:avg_generation_throughput_toks_per_s', 0),

                # Preemption
                num_preemptions=int(metrics.get('vllm:num_preemptions_total', 0)),

                # Batch
                num_batched_tokens=int(metrics.get('vllm:num_batched_tokens', 0)),
            )

        except Exception as e:
            return None

    def _sample_loop(self):
        """采样循环"""
        while self.running:
            sample = self._fetch_metrics()
            if sample:
                with self.lock:
                    self.samples.append(sample)
            time.sleep(self.sample_interval)

    def start(self):
        """开始采样"""
        self.samples = []
        self.start_time = time.time()
        self.running = True
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()

    def stop(self) -> List[VLLMMetricsSample]:
        """停止采样"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        return self.samples

    def get_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        if not self.samples:
            return {}

        cache_hits = [s.prefix_cache_hit_rate for s in self.samples]
        gpu_usage = [s.gpu_cache_usage_perc for s in self.samples]
        prompt_tput = [s.prompt_throughput_toks_per_s for s in self.samples]
        gen_tput = [s.generation_throughput_toks_per_s for s in self.samples]

        return {
            "num_samples": len(self.samples),
            "duration_ms": self.samples[-1].relative_time_ms if self.samples else 0,

            # Cache
            "avg_prefix_cache_hit_rate": sum(cache_hits) / len(cache_hits) if cache_hits else 0,
            "max_prefix_cache_hit_rate": max(cache_hits) if cache_hits else 0,
            "avg_gpu_cache_usage": sum(gpu_usage) / len(gpu_usage) if gpu_usage else 0,
            "max_gpu_cache_usage": max(gpu_usage) if gpu_usage else 0,

            # Throughput
            "avg_prompt_throughput": sum(prompt_tput) / len(prompt_tput) if prompt_tput else 0,
            "max_prompt_throughput": max(prompt_tput) if prompt_tput else 0,
            "avg_generation_throughput": sum(gen_tput) / len(gen_tput) if gen_tput else 0,
            "max_generation_throughput": max(gen_tput) if gen_tput else 0,

            # Preemption
            "total_preemptions": self.samples[-1].num_preemptions if self.samples else 0,
        }


def run_with_metrics(question: str, collector: VLLMMetricsCollector, persist_dir: str, collection_name: str, k: int, max_hops: int) -> Dict[str, Any]:
    """运行查询并收集指标"""
    from Rag.hop2_rag import run_hop2_rag

    start = time.time()
    result = run_hop2_rag(
        question=question,
        persist_dir=persist_dir,
        collection_name=collection_name,
        k=k,
        max_hops=max_hops
    )
    elapsed = time.time() - start

    return {
        "question": question,
        "elapsed_ms": elapsed * 1000,
        "answer": result.get("answer", "")[:100],
        "metadata": result.get("metadata", {}),
    }


def plot_cache_metrics(samples: List[VLLMMetricsSample], output_path: str):
    """绘制缓存指标图"""
    if not HAS_MATPLOTLIB or not samples:
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    times = [s.relative_time_ms / 1000 for s in samples]  # 转换为秒

    # 上图：Cache 命中率
    ax1 = axes[0]
    cache_hits = [s.prefix_cache_hit_rate * 100 for s in samples]
    ax1.plot(times, cache_hits, color='forestgreen', linewidth=2, label='Prefix Cache Hit Rate')
    ax1.fill_between(times, cache_hits, alpha=0.3, color='forestgreen')
    ax1.set_ylabel('Hit Rate (%)', fontsize=12)
    ax1.set_title('vLLM Prefix Cache Hit Rate Over Time', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)

    # 下图：GPU Cache 使用率
    ax2 = axes[1]
    gpu_usage = [s.gpu_cache_usage_perc * 100 for s in samples]
    ax2.plot(times, gpu_usage, color='coral', linewidth=2, label='GPU Cache Usage')
    ax2.fill_between(times, gpu_usage, alpha=0.3, color='coral')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Usage (%)', fontsize=12)
    ax2.set_title('GPU KV Cache Usage Over Time', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved cache metrics: {output_path}")


def plot_throughput(samples: List[VLLMMetricsSample], output_path: str):
    """绘制吞吐量图"""
    if not HAS_MATPLOTLIB or not samples:
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    times = [s.relative_time_ms / 1000 for s in samples]
    prompt_tput = [s.prompt_throughput_toks_per_s for s in samples]
    gen_tput = [s.generation_throughput_toks_per_s for s in samples]

    ax.plot(times, prompt_tput, color='steelblue', linewidth=2, label='Prompt Throughput')
    ax.plot(times, gen_tput, color='coral', linewidth=2, label='Generation Throughput')

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Tokens/s', fontsize=12)
    ax.set_title('vLLM Throughput Over Time', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved throughput: {output_path}")


def plot_request_states(samples: List[VLLMMetricsSample], output_path: str):
    """绘制请求状态图"""
    if not HAS_MATPLOTLIB or not samples:
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    times = [s.relative_time_ms / 1000 for s in samples]
    running = [s.num_requests_running for s in samples]
    waiting = [s.num_requests_waiting for s in samples]
    swapped = [s.num_requests_swapped for s in samples]

    ax.stackplot(times, running, waiting, swapped,
                labels=['Running', 'Waiting', 'Swapped'],
                colors=['forestgreen', 'gold', 'coral'],
                alpha=0.7)

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Number of Requests', fontsize=12)
    ax.set_title('vLLM Request States Over Time', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved request states: {output_path}")


def compare_with_without_prefix_caching(questions: List[str], persist_dir: str, collection_name: str, k: int, max_hops: int) -> Dict[str, Any]:
    """对比有无 prefix caching 的效果（需要两次运行）"""
    print("\n[INFO] To compare with/without prefix caching:")
    print("  1. Run vLLM without --enable-prefix-caching, run this script")
    print("  2. Run vLLM with --enable-prefix-caching, run this script again")
    print("  3. Compare the results manually")

    return {}


def main():
    parser = argparse.ArgumentParser(
        description="vLLM Metrics Observation for Hop2Rag"
    )
    parser.add_argument("--limit", type=int, default=10, help="Number of questions")
    parser.add_argument("--k", type=int, default=10, help="Documents per hop")
    parser.add_argument("--max-hops", type=int, default=3, help="Maximum hops")
    parser.add_argument("--vllm-endpoint", type=str, default="http://localhost:8000", help="vLLM endpoint")
    parser.add_argument("--sample-interval", type=int, default=100, help="Sample interval in ms")
    parser.add_argument("--skip-plots", action="store_true", help="Skip visualization")

    args = parser.parse_args()

    # 检查环境变量
    if not PERSIST_DIR:
        print("[ERROR] Please set AGRAG_PERSIST_DIR environment variable")
        sys.exit(1)

    print("=" * 70)
    print("vLLM Metrics Observation")
    print("=" * 70)
    print(f"Questions:        {args.limit}")
    print(f"K:                {args.k}")
    print(f"Max Hops:         {args.max_hops}")
    print(f"vLLM Endpoint:    {args.vllm_endpoint}")
    print(f"Sample Interval:  {args.sample_interval} ms")
    print(f"PERSIST_DIR:      {PERSIST_DIR}")
    print("=" * 70)
    print()

    # 创建输出目录
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 创建指标采集器
    collector = VLLMMetricsCollector(
        vllm_endpoint=args.vllm_endpoint,
        sample_interval_ms=args.sample_interval
    )

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
    questions = (sample_questions * ((args.limit // len(sample_questions)) + 1))[:args.limit]

    # 开始采集
    print("Starting metrics collection...")
    collector.start()

    # 运行查询
    query_results = []
    for i, question in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] Processing: {question[:50]}...")

        try:
            result = run_with_metrics(
                question=question,
                collector=collector,
                persist_dir=PERSIST_DIR,
                collection_name=COLLECTION_NAME,
                k=args.k,
                max_hops=args.max_hops
            )
            query_results.append(result)
            print(f"  Elapsed: {result['elapsed_ms']:.0f} ms")

        except Exception as e:
            print(f"  [ERROR] {e}")
            query_results.append({"question": question, "error": str(e)})

    # 停止采集
    samples = collector.stop()
    summary = collector.get_summary()

    print(f"\n✓ Collected {len(samples)} metric samples")

    # 保存结果
    output_json = RESULTS_DIR / "metrics_timeline.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": summary,
            "query_results": query_results,
            "samples": [asdict(s) for s in samples[-500:]],  # 只保存最后500个样本
        }, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved results: {output_json}")

    # 绘图
    if not args.skip_plots and HAS_MATPLOTLIB and samples:
        print("\nGenerating visualizations...")
        plot_cache_metrics(samples, str(RESULTS_DIR / "cache_metrics.png"))
        plot_throughput(samples, str(RESULTS_DIR / "throughput.png"))
        plot_request_states(samples, str(RESULTS_DIR / "request_states.png"))

    # 打印摘要
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Requests Processed:        {len(query_results)}")
    print(f"Metrics Samples:           {summary.get('num_samples', 0)}")
    print(f"Duration:                  {summary.get('duration_ms', 0):.0f} ms")
    print()
    print("Cache Metrics:")
    print(f"  Avg Prefix Cache Hit:    {summary.get('avg_prefix_cache_hit_rate', 0):.1%}")
    print(f"  Max Prefix Cache Hit:    {summary.get('max_prefix_cache_hit_rate', 0):.1%}")
    print(f"  Avg GPU Cache Usage:     {summary.get('avg_gpu_cache_usage', 0):.1%}")
    print()
    print("Throughput:")
    print(f"  Avg Prompt Throughput:   {summary.get('avg_prompt_throughput', 0):.0f} tokens/s")
    print(f"  Avg Gen Throughput:      {summary.get('avg_generation_throughput', 0):.0f} tokens/s")
    print()
    print("Preemption:")
    print(f"  Total Preemptions:       {summary.get('total_preemptions', 0)}")
    print()

    # 建议
    hit_rate = summary.get('avg_prefix_cache_hit_rate', 0)
    if hit_rate < 0.1:
        print("Observations:")
        print("  - Low prefix cache hit rate detected")
        print("  - Consider:")
        print("    * Verify vLLM started with --enable-prefix-caching")
        print("    * Redesign prompts to maximize prefix sharing")
        print("    * Use consistent system prompts across requests")
    elif hit_rate > 0.5:
        print("Observations:")
        print("  - Good prefix cache hit rate!")
        print("  - Agentic RAG prompts have significant prefix overlap")
        print("  - KV Cache reuse is effective")

    print()
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
