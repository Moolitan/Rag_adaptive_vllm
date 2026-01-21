#!/usr/bin/env python3
"""
实验五：上下文增长影响分析

目标：测量随着跳数增加，累积的文档上下文如何影响延迟和内存。

观察点：
- 上下文长度与推理延迟的关系（线性？超线性？）
- 上下文长度与内存占用的关系
- 不同跳数下的 token 累积模式
- Attention 计算复杂度的实际影响

输出：
- results/hop2rag/context_growth/growth_analysis.json
- results/hop2rag/context_growth/context_vs_latency.png
- results/hop2rag/context_growth/context_vs_memory.png
- results/hop2rag/context_growth/token_accumulation.png
"""

import argparse
import json
import sys
import os
import time
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional
from collections import defaultdict

# Setup paths
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# Check for CUDA
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False
    print("[WARN] PyTorch not found")

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARN] matplotlib not found, visualization disabled")

# Environment variables
PERSIST_DIR = os.environ.get("AGRAG_PERSIST_DIR", "")
COLLECTION_NAME = os.environ.get("AGRAG_COLLECTION_NAME", "hotpot_fullwiki")

# Results directory
RESULTS_DIR = ROOT / "tests" / "results" / "hop2rag" / "context_growth"


@dataclass
class HopMeasurement:
    """单跳测量数据"""
    hop: int
    context_tokens: int
    context_chars: int
    num_documents: int

    # 延迟
    decompose_latency_ms: float = 0
    extract_clues_latency_ms: float = 0
    decide_latency_ms: float = 0
    total_hop_latency_ms: float = 0

    # 内存
    memory_before_mb: float = 0
    memory_after_mb: float = 0
    memory_peak_mb: float = 0


@dataclass
class RequestMeasurement:
    """单请求测量数据"""
    request_id: str
    question: str
    total_hops: int
    total_latency_ms: float

    # 最终生成阶段
    final_context_tokens: int = 0
    generate_latency_ms: float = 0

    # 每跳数据
    hop_measurements: List[HopMeasurement] = field(default_factory=list)


class ContextGrowthAnalyzer:
    """上下文增长分析器"""

    def __init__(self):
        self.measurements: List[RequestMeasurement] = []

    def estimate_tokens(self, text: str) -> int:
        """估算 token 数量（简单方法：字符数/4）"""
        return len(text) // 4

    def run_instrumented_query(
        self,
        question: str,
        request_id: str,
        persist_dir: str,
        collection_name: str,
        k: int,
        max_hops: int
    ) -> RequestMeasurement:
        """运行带详细测量的查询"""
        from Rag.hop2_rag import get_hop2_rag_app

        app = get_hop2_rag_app()

        inputs = {
            "question": question,
            "custom_retriever_config": {
                "persist_dir": persist_dir,
                "collection_name": collection_name,
                "k": str(k),
                "max_hops": str(max_hops)
            }
        }

        measurement = RequestMeasurement(
            request_id=request_id,
            question=question,
            total_hops=0,
            total_latency_ms=0,
        )

        hop_data = defaultdict(lambda: {
            "context_tokens": 0,
            "context_chars": 0,
            "num_documents": 0,
            "latencies": {},
            "memory": {}
        })

        current_hop = 0
        start_time = time.time()

        # 使用 stream 模式获取中间状态
        for event in app.stream(inputs, stream_mode="updates"):
            for node_name, update in event.items():
                node_start = time.time()

                # 获取内存（如果可用）
                mem_before = 0
                if HAS_CUDA:
                    mem_before = torch.cuda.memory_allocated() / 1024**2

                # 提取 hop 信息
                if isinstance(update, dict):
                    new_hop = update.get("current_hop", current_hop)
                    if new_hop != current_hop:
                        current_hop = new_hop

                    # 记录文档信息
                    docs = update.get("all_graded_documents", [])
                    if docs:
                        total_chars = sum(len(d.page_content) for d in docs)
                        hop_data[current_hop]["context_chars"] = total_chars
                        hop_data[current_hop]["context_tokens"] = self.estimate_tokens(
                            "".join(d.page_content for d in docs)
                        )
                        hop_data[current_hop]["num_documents"] = len(docs)

                node_latency = (time.time() - node_start) * 1000

                # 记录 LLM 节点延迟
                if "decompose" in node_name:
                    hop_data[current_hop]["latencies"]["decompose"] = node_latency
                elif "extract_clues" in node_name:
                    hop_data[current_hop]["latencies"]["extract_clues"] = node_latency
                elif "decide" in node_name:
                    hop_data[current_hop]["latencies"]["decide"] = node_latency
                elif "generate_final" in node_name:
                    measurement.generate_latency_ms = node_latency

                # 记录内存
                if HAS_CUDA:
                    mem_after = torch.cuda.memory_allocated() / 1024**2
                    mem_peak = torch.cuda.max_memory_allocated() / 1024**2
                    hop_data[current_hop]["memory"] = {
                        "before": mem_before,
                        "after": mem_after,
                        "peak": mem_peak
                    }

        total_latency = (time.time() - start_time) * 1000
        measurement.total_latency_ms = total_latency
        measurement.total_hops = current_hop

        # 转换 hop_data 为 HopMeasurement 列表
        for hop, data in sorted(hop_data.items()):
            hop_m = HopMeasurement(
                hop=hop,
                context_tokens=data["context_tokens"],
                context_chars=data["context_chars"],
                num_documents=data["num_documents"],
                decompose_latency_ms=data["latencies"].get("decompose", 0),
                extract_clues_latency_ms=data["latencies"].get("extract_clues", 0),
                decide_latency_ms=data["latencies"].get("decide", 0),
                total_hop_latency_ms=sum(data["latencies"].values()),
                memory_before_mb=data["memory"].get("before", 0),
                memory_after_mb=data["memory"].get("after", 0),
                memory_peak_mb=data["memory"].get("peak", 0),
            )
            measurement.hop_measurements.append(hop_m)

        # 最终上下文 tokens
        if measurement.hop_measurements:
            measurement.final_context_tokens = measurement.hop_measurements[-1].context_tokens

        self.measurements.append(measurement)
        return measurement

    def analyze_growth_pattern(self) -> Dict[str, Any]:
        """分析增长模式"""
        if not self.measurements:
            return {}

        # 收集所有 hop 数据点
        all_points = []
        for m in self.measurements:
            for hop_m in m.hop_measurements:
                if hop_m.context_tokens > 0:
                    all_points.append({
                        "hop": hop_m.hop,
                        "context_tokens": hop_m.context_tokens,
                        "total_latency_ms": hop_m.total_hop_latency_ms,
                        "memory_peak_mb": hop_m.memory_peak_mb,
                    })

        if not all_points:
            return {}

        # 按 context_tokens 排序
        all_points.sort(key=lambda x: x["context_tokens"])

        tokens = [p["context_tokens"] for p in all_points]
        latencies = [p["total_latency_ms"] for p in all_points]
        memories = [p["memory_peak_mb"] for p in all_points]

        # 线性回归分析
        result = {
            "num_data_points": len(all_points),
            "token_range": [min(tokens), max(tokens)],
            "latency_range_ms": [min(latencies), max(latencies)],
        }

        if HAS_MATPLOTLIB and len(tokens) > 2:
            # Token vs Latency 回归
            slope, intercept, r_value, p_value, std_err = stats.linregress(tokens, latencies)
            result["token_latency_correlation"] = {
                "slope_ms_per_token": slope,
                "intercept_ms": intercept,
                "r_squared": r_value ** 2,
                "p_value": p_value,
            }

            # 判断是否超线性
            # 尝试二次拟合
            if len(tokens) > 3:
                coeffs = np.polyfit(tokens, latencies, 2)
                result["quadratic_coefficient"] = coeffs[0]
                result["is_superlinear"] = coeffs[0] > 0.0001  # 阈值判断

        return result

    def get_summary(self) -> Dict[str, Any]:
        """获取分析摘要"""
        if not self.measurements:
            return {}

        total_requests = len(self.measurements)
        avg_hops = sum(m.total_hops for m in self.measurements) / total_requests
        avg_latency = sum(m.total_latency_ms for m in self.measurements) / total_requests
        avg_final_tokens = sum(m.final_context_tokens for m in self.measurements) / total_requests

        # 按跳数统计
        by_hop = defaultdict(list)
        for m in self.measurements:
            for hop_m in m.hop_measurements:
                by_hop[hop_m.hop].append(hop_m)

        hop_stats = {}
        for hop, measurements in sorted(by_hop.items()):
            hop_stats[f"hop_{hop}"] = {
                "avg_context_tokens": sum(h.context_tokens for h in measurements) / len(measurements),
                "avg_latency_ms": sum(h.total_hop_latency_ms for h in measurements) / len(measurements),
                "avg_memory_peak_mb": sum(h.memory_peak_mb for h in measurements) / len(measurements),
            }

        return {
            "total_requests": total_requests,
            "avg_hops": avg_hops,
            "avg_total_latency_ms": avg_latency,
            "avg_final_context_tokens": avg_final_tokens,
            "by_hop": hop_stats,
            "growth_analysis": self.analyze_growth_pattern(),
        }


def plot_context_vs_latency(measurements: List[RequestMeasurement], output_path: str):
    """绘制上下文 vs 延迟图"""
    if not HAS_MATPLOTLIB or not measurements:
        return

    # 收集所有数据点
    tokens = []
    latencies = []
    hops = []

    for m in measurements:
        for hop_m in m.hop_measurements:
            if hop_m.context_tokens > 0 and hop_m.total_hop_latency_ms > 0:
                tokens.append(hop_m.context_tokens)
                latencies.append(hop_m.total_hop_latency_ms)
                hops.append(hop_m.hop)

    if not tokens:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # 散点图，颜色表示 hop
    scatter = ax.scatter(tokens, latencies, c=hops, cmap='viridis',
                         s=80, alpha=0.7, edgecolors='black', linewidths=0.5)

    # 拟合线
    if len(tokens) > 2:
        z = np.polyfit(tokens, latencies, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(tokens), max(tokens), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Linear fit (slope={z[0]:.4f})')

        # 二次拟合
        z2 = np.polyfit(tokens, latencies, 2)
        p2 = np.poly1d(z2)
        ax.plot(x_line, p2(x_line), 'g--', linewidth=2, alpha=0.7,
                label=f'Quadratic fit (a={z2[0]:.6f})')

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Hop Number', fontsize=11)

    ax.set_xlabel('Context Tokens', fontsize=12)
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('Context Length vs LLM Latency\n(Is it linear or superlinear?)', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved context vs latency: {output_path}")


def plot_context_vs_memory(measurements: List[RequestMeasurement], output_path: str):
    """绘制上下文 vs 内存图"""
    if not HAS_MATPLOTLIB or not measurements:
        return

    tokens = []
    memories = []

    for m in measurements:
        for hop_m in m.hop_measurements:
            if hop_m.context_tokens > 0 and hop_m.memory_peak_mb > 0:
                tokens.append(hop_m.context_tokens)
                memories.append(hop_m.memory_peak_mb)

    if not tokens:
        print("[WARN] No memory data available (CUDA not enabled?)")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(tokens, memories, s=80, alpha=0.7, color='coral',
               edgecolors='black', linewidths=0.5)

    # 拟合线
    if len(tokens) > 2:
        z = np.polyfit(tokens, memories, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(tokens), max(tokens), 100)
        ax.plot(x_line, p(x_line), 'b--', linewidth=2,
                label=f'Linear fit (slope={z[0]:.4f} MB/token)')

    ax.set_xlabel('Context Tokens', fontsize=12)
    ax.set_ylabel('Peak Memory (MB)', fontsize=12)
    ax.set_title('Context Length vs GPU Memory\n(KV Cache growth pattern)', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved context vs memory: {output_path}")


def plot_token_accumulation(measurements: List[RequestMeasurement], output_path: str):
    """绘制 token 累积模式图"""
    if not HAS_MATPLOTLIB or not measurements:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：每跳的 token 增量
    ax1 = axes[0]
    by_hop = defaultdict(list)

    for m in measurements:
        prev_tokens = 0
        for hop_m in m.hop_measurements:
            delta = hop_m.context_tokens - prev_tokens
            by_hop[hop_m.hop].append(delta)
            prev_tokens = hop_m.context_tokens

    hops = sorted(by_hop.keys())
    avg_deltas = [np.mean(by_hop[h]) for h in hops]
    std_deltas = [np.std(by_hop[h]) for h in hops]

    ax1.bar(hops, avg_deltas, yerr=std_deltas, capsize=5,
            color='steelblue', alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Hop', fontsize=12)
    ax1.set_ylabel('Token Increment', fontsize=12)
    ax1.set_title('Token Accumulation per Hop', fontsize=13)
    ax1.grid(True, alpha=0.3, axis='y')

    # 右图：累积 token vs hop
    ax2 = axes[1]

    for i, m in enumerate(measurements[:5]):  # 只显示前5个请求
        hops = [hop_m.hop for hop_m in m.hop_measurements]
        tokens = [hop_m.context_tokens for hop_m in m.hop_measurements]
        ax2.plot(hops, tokens, marker='o', linewidth=2, alpha=0.7, label=f'Request {i+1}')

    ax2.set_xlabel('Hop', fontsize=12)
    ax2.set_ylabel('Cumulative Context Tokens', fontsize=12)
    ax2.set_title('Context Growth Across Hops\n(Sample of 5 requests)', fontsize=13)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved token accumulation: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Context Growth Impact Analysis for Hop2Rag"
    )
    parser.add_argument("--limit", type=int, default=10, help="Number of questions")
    parser.add_argument("--k", type=int, default=10, help="Documents per hop")
    parser.add_argument("--max-hops", type=int, default=5, help="Maximum hops")
    parser.add_argument("--skip-plots", action="store_true", help="Skip visualization")

    args = parser.parse_args()

    # 检查环境变量
    if not PERSIST_DIR:
        print("[ERROR] Please set AGRAG_PERSIST_DIR environment variable")
        sys.exit(1)

    print("=" * 70)
    print("Context Growth Impact Analysis")
    print("=" * 70)
    print(f"Questions:     {args.limit}")
    print(f"K:             {args.k}")
    print(f"Max Hops:      {args.max_hops}")
    print(f"CUDA:          {HAS_CUDA}")
    print(f"PERSIST_DIR:   {PERSIST_DIR}")
    print("=" * 70)
    print()

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
    questions = (sample_questions * ((args.limit // len(sample_questions)) + 1))[:args.limit]

    # 创建分析器
    analyzer = ContextGrowthAnalyzer()

    # 运行查询
    for i, question in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] Processing: {question[:50]}...")

        try:
            measurement = analyzer.run_instrumented_query(
                question=question,
                request_id=f"req_{i:04d}",
                persist_dir=PERSIST_DIR,
                collection_name=COLLECTION_NAME,
                k=args.k,
                max_hops=args.max_hops
            )

            print(f"  Hops: {measurement.total_hops}")
            print(f"  Final Context: {measurement.final_context_tokens} tokens")
            print(f"  Total Latency: {measurement.total_latency_ms:.0f} ms")

        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()

    # 获取摘要
    summary = analyzer.get_summary()

    # 保存结果
    output_json = RESULTS_DIR / "growth_analysis.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        result = {
            "summary": summary,
            "measurements": [
                {
                    "request_id": m.request_id,
                    "question": m.question,
                    "total_hops": m.total_hops,
                    "total_latency_ms": m.total_latency_ms,
                    "final_context_tokens": m.final_context_tokens,
                    "generate_latency_ms": m.generate_latency_ms,
                    "hop_measurements": [asdict(h) for h in m.hop_measurements]
                }
                for m in analyzer.measurements
            ]
        }
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Saved results: {output_json}")

    # 绘图
    if not args.skip_plots and HAS_MATPLOTLIB:
        print("\nGenerating visualizations...")
        plot_context_vs_latency(analyzer.measurements, str(RESULTS_DIR / "context_vs_latency.png"))
        plot_context_vs_memory(analyzer.measurements, str(RESULTS_DIR / "context_vs_memory.png"))
        plot_token_accumulation(analyzer.measurements, str(RESULTS_DIR / "token_accumulation.png"))

    # 打印摘要
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Requests Analyzed:      {summary.get('total_requests', 0)}")
    print(f"Avg Hops:               {summary.get('avg_hops', 0):.1f}")
    print(f"Avg Total Latency:      {summary.get('avg_total_latency_ms', 0):.0f} ms")
    print(f"Avg Final Context:      {summary.get('avg_final_context_tokens', 0):.0f} tokens")
    print()

    growth = summary.get("growth_analysis", {})
    if growth:
        print("Growth Pattern Analysis:")
        token_latency = growth.get("token_latency_correlation", {})
        if token_latency:
            print(f"  Slope:        {token_latency.get('slope_ms_per_token', 0):.4f} ms/token")
            print(f"  R-squared:    {token_latency.get('r_squared', 0):.3f}")
            print(f"  Is Superlinear: {growth.get('is_superlinear', False)}")

    print()
    print("By Hop Statistics:")
    for hop_name, hop_stats in summary.get("by_hop", {}).items():
        print(f"  {hop_name}:")
        print(f"    Avg Tokens:  {hop_stats.get('avg_context_tokens', 0):.0f}")
        print(f"    Avg Latency: {hop_stats.get('avg_latency_ms', 0):.0f} ms")

    print()
    print("Implications:")
    if growth.get("is_superlinear", False):
        print("  - Superlinear growth detected!")
        print("  - Attention computation becomes bottleneck at high context lengths")
        print("  - Consider: context compression, sliding window attention")
    else:
        print("  - Linear growth pattern")
        print("  - Latency scales predictably with context length")

    print()
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
