#!/usr/bin/env python3
"""
实验三：检索-生成流水线重叠分析

目标：分析检索（CPU/IO 密集）和生成（GPU 密集）操作的时间分布，
     评估流水线并行的可能性。

观察点：
- 检索阶段 GPU 空闲率
- 生成阶段 CPU 空闲率
- 理论可重叠时间比例
- 预取（Prefetch）优化的潜在收益

输出：
- results/hop2rag/pipeline/timeline_gantt.png
- results/hop2rag/pipeline/overlap_analysis.json
- results/hop2rag/pipeline/resource_utilization.png
"""

import argparse
import json
import sys
import os
import time
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from enum import Enum

# Setup paths
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARN] matplotlib not found, visualization disabled")

# Environment variables
PERSIST_DIR = os.environ.get("AGRAG_PERSIST_DIR", "")
COLLECTION_NAME = os.environ.get("AGRAG_COLLECTION_NAME", "hotpot_fullwiki")

# Results directory
RESULTS_DIR = ROOT / "tests" / "results" / "hop2rag" / "pipeline"


class ResourceType(Enum):
    """资源类型"""
    CPU = "cpu"
    GPU = "gpu"
    IO = "io"
    MIXED = "mixed"


@dataclass
class NodeExecution:
    """节点执行记录"""
    node_name: str
    hop: int
    start_time: float
    end_time: float
    duration_ms: float
    resource_type: ResourceType

    @property
    def uses_gpu(self) -> bool:
        return self.resource_type in [ResourceType.GPU, ResourceType.MIXED]

    @property
    def uses_cpu(self) -> bool:
        return self.resource_type in [ResourceType.CPU, ResourceType.IO, ResourceType.MIXED]


@dataclass
class OverlapOpportunity:
    """重叠优化机会"""
    node_a: str  # 当前执行的节点
    node_b: str  # 可并行的节点
    potential_overlap_ms: float
    resource_a: ResourceType
    resource_b: ResourceType
    feasibility: str  # "high", "medium", "low"


# 节点资源分类
NODE_RESOURCE_MAP = {
    # LLM 节点 - GPU 密集
    "decompose": ResourceType.GPU,
    "extract_clues": ResourceType.GPU,
    "decide": ResourceType.GPU,
    "generate_final": ResourceType.GPU,

    # 检索节点 - CPU/IO 密集
    "retrieve_hop": ResourceType.IO,
    "grade_reranker": ResourceType.CPU,  # TF-IDF 在 CPU
    "filter_docs": ResourceType.CPU,

    # 其他节点
    "initialize": ResourceType.CPU,
    "accumulate": ResourceType.CPU,
    "extract_supporting_facts_fast": ResourceType.CPU,
    "finalize": ResourceType.CPU,
}


class PipelineAnalyzer:
    """流水线分析器"""

    def __init__(self):
        self.executions: List[NodeExecution] = []
        self.request_start_time: float = 0

    def record_execution(
        self,
        node_name: str,
        hop: int,
        start_time: float,
        end_time: float
    ):
        """记录节点执行"""
        # 获取资源类型
        resource_type = ResourceType.CPU
        for key, rt in NODE_RESOURCE_MAP.items():
            if key in node_name:
                resource_type = rt
                break

        exec_record = NodeExecution(
            node_name=node_name,
            hop=hop,
            start_time=start_time,
            end_time=end_time,
            duration_ms=(end_time - start_time) * 1000,
            resource_type=resource_type,
        )
        self.executions.append(exec_record)

    def analyze_resource_utilization(self) -> Dict[str, Any]:
        """分析资源利用率"""
        if not self.executions:
            return {}

        total_time = self.executions[-1].end_time - self.executions[0].start_time
        total_time_ms = total_time * 1000

        # 计算各资源占用时间
        gpu_time = sum(e.duration_ms for e in self.executions if e.uses_gpu)
        cpu_time = sum(e.duration_ms for e in self.executions if e.uses_cpu)
        io_time = sum(e.duration_ms for e in self.executions if e.resource_type == ResourceType.IO)

        # GPU 空闲时间（非 GPU 节点执行期间）
        non_gpu_time = sum(e.duration_ms for e in self.executions if not e.uses_gpu)

        return {
            "total_time_ms": total_time_ms,
            "gpu_active_time_ms": gpu_time,
            "cpu_active_time_ms": cpu_time,
            "io_time_ms": io_time,
            "gpu_idle_time_ms": non_gpu_time,
            "gpu_utilization": gpu_time / total_time_ms if total_time_ms > 0 else 0,
            "cpu_utilization": cpu_time / total_time_ms if total_time_ms > 0 else 0,
        }

    def find_overlap_opportunities(self) -> List[OverlapOpportunity]:
        """找到可重叠的机会"""
        opportunities = []

        for i, exec_a in enumerate(self.executions):
            # 只考虑 CPU/IO 节点，因为它们不占用 GPU
            if exec_a.uses_gpu:
                continue

            # 找下一个 GPU 节点
            for exec_b in self.executions[i+1:]:
                if not exec_b.uses_gpu:
                    continue

                # exec_a 是 CPU/IO 节点，exec_b 是 GPU 节点
                # 理论上可以并行

                # 检查是否有数据依赖
                # 简化假设：同一 hop 内有依赖，跨 hop 可能可以预取
                if exec_a.hop == exec_b.hop:
                    feasibility = "low"  # 同 hop 可能有数据依赖
                elif exec_a.hop < exec_b.hop:
                    feasibility = "high"  # 可以预取下一 hop 的数据
                else:
                    feasibility = "medium"

                opportunities.append(OverlapOpportunity(
                    node_a=f"{exec_a.node_name}@hop{exec_a.hop}",
                    node_b=f"{exec_b.node_name}@hop{exec_b.hop}",
                    potential_overlap_ms=min(exec_a.duration_ms, exec_b.duration_ms),
                    resource_a=exec_a.resource_type,
                    resource_b=exec_b.resource_type,
                    feasibility=feasibility,
                ))
                break  # 只找最近的 GPU 节点

        return opportunities

    def calculate_prefetch_potential(self) -> Dict[str, Any]:
        """计算预取优化的潜在收益"""
        # 找所有 retrieve 节点
        retrieve_nodes = [e for e in self.executions if "retrieve" in e.node_name]

        if len(retrieve_nodes) < 2:
            return {"potential_savings_ms": 0, "details": []}

        # 计算如果能在上一个 LLM 节点执行时预取下一个 hop 的文档
        savings = []
        for i, retrieve in enumerate(retrieve_nodes[1:], 1):  # 从第二个开始
            # 找前一个 LLM 节点
            prev_llm = None
            for e in reversed(self.executions):
                if e.end_time <= retrieve.start_time and e.uses_gpu:
                    prev_llm = e
                    break

            if prev_llm:
                # 如果 LLM 节点执行时间 >= retrieve 时间，可以完全隐藏
                overlap = min(prev_llm.duration_ms, retrieve.duration_ms)
                savings.append({
                    "retrieve_node": f"{retrieve.node_name}@hop{retrieve.hop}",
                    "llm_node": f"{prev_llm.node_name}@hop{prev_llm.hop}",
                    "retrieve_time_ms": retrieve.duration_ms,
                    "llm_time_ms": prev_llm.duration_ms,
                    "potential_overlap_ms": overlap,
                })

        total_savings = sum(s["potential_overlap_ms"] for s in savings)

        return {
            "potential_savings_ms": total_savings,
            "num_opportunities": len(savings),
            "details": savings,
        }


def run_pipeline_analysis(question: str, persist_dir: str, collection_name: str, k: int, max_hops: int) -> PipelineAnalyzer:
    """运行流水线分析"""
    from Rag.hop2_rag import get_hop2_rag_app

    analyzer = PipelineAnalyzer()
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

    analyzer.request_start_time = time.time()

    # 使用 stream 模式记录每个节点的执行时间
    last_time = time.time()
    last_node = ""
    last_hop = 0

    for event in app.stream(inputs, stream_mode="updates"):
        current_time = time.time()

        for node_name, update in event.items():
            # 记录上一个节点的结束
            if last_node:
                analyzer.record_execution(
                    node_name=last_node,
                    hop=last_hop,
                    start_time=last_time,
                    end_time=current_time
                )

            # 更新当前节点
            last_node = node_name
            last_hop = update.get("current_hop", 0) if isinstance(update, dict) else 0
            last_time = current_time

    # 记录最后一个节点
    if last_node:
        analyzer.record_execution(
            node_name=last_node,
            hop=last_hop,
            start_time=last_time,
            end_time=time.time()
        )

    return analyzer


def plot_gantt_chart(executions: List[NodeExecution], output_path: str):
    """绘制甘特图"""
    if not HAS_MATPLOTLIB or not executions:
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    # 颜色映射
    color_map = {
        ResourceType.GPU: 'coral',
        ResourceType.CPU: 'steelblue',
        ResourceType.IO: 'forestgreen',
        ResourceType.MIXED: 'purple',
    }

    # 计算相对时间
    base_time = executions[0].start_time

    # 绘制每个节点
    y_labels = []
    for i, exe in enumerate(executions):
        start_rel = (exe.start_time - base_time) * 1000
        duration = exe.duration_ms

        label = f"{exe.node_name}\n(hop {exe.hop})"
        y_labels.append(label)

        color = color_map.get(exe.resource_type, 'gray')
        ax.barh(i, duration, left=start_rel, height=0.6, color=color, alpha=0.8, edgecolor='black')

        # 添加时间标注
        ax.text(start_rel + duration / 2, i, f'{duration:.0f}ms',
               ha='center', va='center', fontsize=8, color='white', fontweight='bold')

    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_title('Pipeline Execution Timeline (Gantt Chart)\nOpportunity: Parallel CPU/IO during GPU idle', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')

    # 图例
    legend_patches = [
        mpatches.Patch(color='coral', label='GPU (LLM)'),
        mpatches.Patch(color='steelblue', label='CPU'),
        mpatches.Patch(color='forestgreen', label='I/O (Retrieval)'),
    ]
    ax.legend(handles=legend_patches, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved Gantt chart: {output_path}")


def plot_resource_utilization(utilization: Dict[str, Any], output_path: str):
    """绘制资源利用率图"""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：时间分布饼图
    ax1 = axes[0]
    labels = ['GPU Active', 'GPU Idle\n(CPU/IO Active)']
    sizes = [utilization['gpu_active_time_ms'], utilization['gpu_idle_time_ms']]
    colors = ['coral', 'lightgray']
    explode = (0.05, 0)

    wedges, texts, autotexts = ax1.pie(
        sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90,
        textprops={'fontsize': 11}
    )
    ax1.set_title('GPU Time Distribution\n(Idle time = Optimization opportunity)', fontsize=13)

    # 右图：时间线对比
    ax2 = axes[1]
    categories = ['GPU', 'CPU', 'I/O']
    times = [
        utilization['gpu_active_time_ms'],
        utilization['cpu_active_time_ms'],
        utilization['io_time_ms']
    ]

    bars = ax2.barh(categories, times, color=['coral', 'steelblue', 'forestgreen'], alpha=0.8)

    for bar, t in zip(bars, times):
        ax2.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                f'{t:.0f} ms', va='center', fontsize=11)

    ax2.set_xlabel('Time (ms)', fontsize=12)
    ax2.set_title('Resource Active Time\n(CPU/IO can run parallel to GPU)', fontsize=13)
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved resource utilization: {output_path}")


def plot_prefetch_potential(prefetch_data: Dict[str, Any], output_path: str):
    """绘制预取优化潜力图"""
    if not HAS_MATPLOTLIB:
        return

    details = prefetch_data.get("details", [])
    if not details:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制每个预取机会
    x = range(len(details))
    retrieve_times = [d["retrieve_time_ms"] for d in details]
    overlap_times = [d["potential_overlap_ms"] for d in details]

    ax.bar(x, retrieve_times, width=0.4, label='Retrieve Time', color='forestgreen', alpha=0.8)
    ax.bar([i + 0.4 for i in x], overlap_times, width=0.4, label='Can Overlap', color='coral', alpha=0.8)

    ax.set_xticks([i + 0.2 for i in x])
    ax.set_xticklabels([f"Hop {i+1}" for i in x], fontsize=11)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title(f'Prefetch Optimization Potential\nTotal Savings: {prefetch_data["potential_savings_ms"]:.0f} ms', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved prefetch potential: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Overlap Analysis for Hop2Rag"
    )
    parser.add_argument("--limit", type=int, default=5, help="Number of questions to analyze")
    parser.add_argument("--k", type=int, default=10, help="Documents per hop")
    parser.add_argument("--max-hops", type=int, default=3, help="Maximum hops")
    parser.add_argument("--skip-plots", action="store_true", help="Skip visualization")

    args = parser.parse_args()

    # 检查环境变量
    if not PERSIST_DIR:
        print("[ERROR] Please set AGRAG_PERSIST_DIR environment variable")
        sys.exit(1)

    print("=" * 70)
    print("Pipeline Overlap Analysis")
    print("=" * 70)
    print(f"Questions:     {args.limit}")
    print(f"K:             {args.k}")
    print(f"Max Hops:      {args.max_hops}")
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
    ]
    questions = sample_questions[:args.limit]

    all_results = []

    for i, question in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] Analyzing: {question[:50]}...")

        try:
            analyzer = run_pipeline_analysis(
                question=question,
                persist_dir=PERSIST_DIR,
                collection_name=COLLECTION_NAME,
                k=args.k,
                max_hops=args.max_hops
            )

            # 分析
            utilization = analyzer.analyze_resource_utilization()
            opportunities = analyzer.find_overlap_opportunities()
            prefetch = analyzer.calculate_prefetch_potential()

            result = {
                "question": question,
                "num_nodes": len(analyzer.executions),
                "total_time_ms": utilization.get("total_time_ms", 0),
                "utilization": utilization,
                "overlap_opportunities": [
                    {
                        "node_a": o.node_a,
                        "node_b": o.node_b,
                        "potential_overlap_ms": o.potential_overlap_ms,
                        "feasibility": o.feasibility,
                    }
                    for o in opportunities
                ],
                "prefetch_potential": prefetch,
                "executions": [asdict(e) for e in analyzer.executions],
            }

            # 转换 enum
            for exe in result["executions"]:
                exe["resource_type"] = exe["resource_type"].value

            all_results.append(result)

            print(f"  Total Time:    {utilization.get('total_time_ms', 0):.0f} ms")
            print(f"  GPU Active:    {utilization.get('gpu_active_time_ms', 0):.0f} ms ({utilization.get('gpu_utilization', 0):.1%})")
            print(f"  GPU Idle:      {utilization.get('gpu_idle_time_ms', 0):.0f} ms")
            print(f"  Prefetch Savings: {prefetch.get('potential_savings_ms', 0):.0f} ms")

            # 为第一个请求绘图
            if i == 0 and not args.skip_plots and HAS_MATPLOTLIB:
                plot_gantt_chart(analyzer.executions, str(RESULTS_DIR / "timeline_gantt.png"))
                plot_resource_utilization(utilization, str(RESULTS_DIR / "resource_utilization.png"))
                if prefetch.get("details"):
                    plot_prefetch_potential(prefetch, str(RESULTS_DIR / "prefetch_potential.png"))

        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"question": question, "error": str(e)})

    # 保存结果
    output_json = RESULTS_DIR / "overlap_analysis.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Saved results: {output_json}")

    # 汇总
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    valid_results = [r for r in all_results if "error" not in r]
    if valid_results:
        avg_gpu_util = sum(r["utilization"]["gpu_utilization"] for r in valid_results) / len(valid_results)
        avg_idle = sum(r["utilization"]["gpu_idle_time_ms"] for r in valid_results) / len(valid_results)
        avg_prefetch = sum(r["prefetch_potential"]["potential_savings_ms"] for r in valid_results) / len(valid_results)

        print(f"Requests Analyzed:      {len(valid_results)}")
        print(f"Avg GPU Utilization:    {avg_gpu_util:.1%}")
        print(f"Avg GPU Idle Time:      {avg_idle:.0f} ms")
        print(f"Avg Prefetch Savings:   {avg_prefetch:.0f} ms")
        print()

        print("Key Findings:")
        print(f"  - {100 - avg_gpu_util*100:.0f}% of execution time, GPU is idle")
        print(f"  - Prefetch could save ~{avg_prefetch:.0f} ms per request")
        print()

        print("Optimization Strategies:")
        print("  1. Prefetch: Start next-hop retrieval during current LLM inference")
        print("  2. Async Embedding: Compute embeddings while LLM generates")
        print("  3. Pipeline Parallelism: Overlap CPU-bound and GPU-bound stages")

    print()
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
