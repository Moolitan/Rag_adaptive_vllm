#!/usr/bin/env python3
"""
实验二：GPU 内存时序分析

目标：记录一个完整请求生命周期中的 GPU 内存变化，
     识别内存峰值位置和空闲窗口。

观察点：
- 检索阶段 vs 生成阶段的显存占用模式
- 峰值内存出现的节点位置
- 内存空闲窗口（可用于调度优化的时机）
- 内存碎片化情况

输出：
- results/hop2rag/gpu_memory/memory_timeline.json
- results/hop2rag/gpu_memory/memory_timeline.png
- results/hop2rag/gpu_memory/memory_by_node.png
"""

import argparse
import json
import sys
import os
import time
import threading
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
    if HAS_CUDA:
        print(f"[INFO] CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("[WARN] CUDA not available, using mock data")
except ImportError:
    HAS_CUDA = False
    print("[WARN] PyTorch not found, using mock data")

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
RESULTS_DIR = ROOT / "tests" / "results" / "hop2rag" / "gpu_memory"


@dataclass
class MemorySample:
    """单个内存采样点"""
    timestamp: float
    relative_time_ms: float
    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float
    current_node: str = ""
    current_hop: int = -1


@dataclass
class NodeMemoryStats:
    """节点级内存统计"""
    node_name: str
    start_allocated_mb: float
    end_allocated_mb: float
    peak_allocated_mb: float
    duration_ms: float
    memory_delta_mb: float  # end - start


class GPUMemoryProfiler:
    """GPU 内存时序采样器"""

    def __init__(self, sample_interval_ms: int = 10):
        self.sample_interval = sample_interval_ms / 1000
        self.samples: List[MemorySample] = []
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.start_time: float = 0
        self.current_node: str = ""
        self.current_hop: int = -1
        self.lock = threading.Lock()

    def _get_memory_stats(self) -> Dict[str, float]:
        """获取当前 GPU 内存状态"""
        if HAS_CUDA:
            return {
                "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
            }
        else:
            # Mock data for testing without GPU
            import random
            base = 1000 + random.gauss(0, 50)
            return {
                "allocated_mb": base,
                "reserved_mb": base + 500,
                "max_allocated_mb": base + 200,
            }

    def _sample_loop(self):
        """采样循环"""
        while self.running:
            now = time.time()
            stats = self._get_memory_stats()

            with self.lock:
                sample = MemorySample(
                    timestamp=now,
                    relative_time_ms=(now - self.start_time) * 1000,
                    allocated_mb=stats["allocated_mb"],
                    reserved_mb=stats["reserved_mb"],
                    max_allocated_mb=stats["max_allocated_mb"],
                    current_node=self.current_node,
                    current_hop=self.current_hop,
                )
                self.samples.append(sample)

            time.sleep(self.sample_interval)

    def start(self):
        """开始采样"""
        if HAS_CUDA:
            torch.cuda.reset_peak_memory_stats()

        self.samples = []
        self.start_time = time.time()
        self.running = True
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()

    def stop(self) -> List[MemorySample]:
        """停止采样并返回结果"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        return self.samples

    def set_context(self, node_name: str, hop: int):
        """设置当前上下文（用于标记采样点）"""
        with self.lock:
            self.current_node = node_name
            self.current_hop = hop

    def get_node_stats(self) -> List[NodeMemoryStats]:
        """计算每个节点的内存统计"""
        # 按节点分组
        by_node = defaultdict(list)
        for s in self.samples:
            if s.current_node:
                by_node[s.current_node].append(s)

        stats = []
        for node_name, node_samples in by_node.items():
            if not node_samples:
                continue

            allocated = [s.allocated_mb for s in node_samples]
            times = [s.relative_time_ms for s in node_samples]

            stats.append(NodeMemoryStats(
                node_name=node_name,
                start_allocated_mb=allocated[0],
                end_allocated_mb=allocated[-1],
                peak_allocated_mb=max(allocated),
                duration_ms=times[-1] - times[0] if len(times) > 1 else 0,
                memory_delta_mb=allocated[-1] - allocated[0],
            ))

        return stats


class InstrumentedHop2Rag:
    """带内存监控的 Hop2Rag 执行器"""

    def __init__(self, profiler: GPUMemoryProfiler, persist_dir: str, collection_name: str, k: int, max_hops: int):
        self.profiler = profiler
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.k = k
        self.max_hops = max_hops

        # Node timing records
        self.node_times: Dict[str, List[float]] = defaultdict(list)

    def run(self, question: str) -> Dict[str, Any]:
        """运行带内存监控的查询"""
        from Rag.hop2_rag import (
            get_hop2_rag_app,
            get_question_decomposer,
            get_clue_extractor,
            get_hop_decision_chain,
            get_multi_hop_rag_chain,
            get_custom_retriever,
            get_reranker,
            get_sentence_selector,
        )

        # 预热/初始化组件
        self.profiler.set_context("init", -1)
        app = get_hop2_rag_app()

        inputs = {
            "question": question,
            "custom_retriever_config": {
                "persist_dir": self.persist_dir,
                "collection_name": self.collection_name,
                "k": str(self.k),
                "max_hops": str(self.max_hops)
            }
        }

        # 执行（使用 app.invoke，同时监控内存）
        # 由于 LangGraph 内部执行，我们通过 stream 来标记节点
        self.profiler.set_context("invoke_start", 0)

        result = None
        try:
            # 使用 stream 模式可以看到中间状态
            for event in app.stream(inputs, stream_mode="updates"):
                # event 是 {node_name: state_update} 格式
                for node_name, update in event.items():
                    # 提取 hop 信息
                    hop = update.get("current_hop", 0) if isinstance(update, dict) else 0
                    self.profiler.set_context(node_name, hop)

                    # 记录节点开始
                    start = time.time()

                    # 等待一小段时间让采样器捕获
                    time.sleep(0.05)

                    # 记录节点时间
                    self.node_times[node_name].append(time.time() - start)

                    # 保存最终结果
                    if node_name == "finalize":
                        result = update

        except Exception as e:
            print(f"[ERROR] Execution failed: {e}")
            import traceback
            traceback.print_exc()
            result = {"error": str(e)}

        self.profiler.set_context("done", -1)
        return result or {}


def plot_memory_timeline(samples: List[MemorySample], output_path: str):
    """绘制内存时序图"""
    if not HAS_MATPLOTLIB or not samples:
        return

    times = [s.relative_time_ms for s in samples]
    allocated = [s.allocated_mb for s in samples]
    reserved = [s.reserved_mb for s in samples]

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(times, allocated, label='Allocated', color='steelblue', linewidth=2)
    ax.plot(times, reserved, label='Reserved', color='coral', linewidth=2, alpha=0.7)

    # 标记节点变化点
    prev_node = ""
    node_changes = []
    for s in samples:
        if s.current_node != prev_node and s.current_node:
            node_changes.append((s.relative_time_ms, s.allocated_mb, s.current_node, s.current_hop))
            prev_node = s.current_node

    # 绘制节点边界
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, (t, mem, node, hop) in enumerate(node_changes):
        ax.axvline(x=t, color=colors[i % 10], linestyle='--', alpha=0.5, linewidth=1)
        ax.annotate(f'{node}\n(hop {hop})',
                   xy=(t, mem), xytext=(5, 10),
                   textcoords='offset points',
                   fontsize=8, rotation=45,
                   color=colors[i % 10])

    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Memory (MB)', fontsize=12)
    ax.set_title('GPU Memory Timeline During Hop2Rag Execution', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # 添加峰值标记
    peak_idx = np.argmax(allocated)
    ax.scatter([times[peak_idx]], [allocated[peak_idx]], color='red', s=100, zorder=5, marker='*')
    ax.annotate(f'Peak: {allocated[peak_idx]:.0f} MB',
               xy=(times[peak_idx], allocated[peak_idx]),
               xytext=(10, -20), textcoords='offset points',
               fontsize=10, color='red')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved memory timeline: {output_path}")


def plot_memory_by_node(node_stats: List[NodeMemoryStats], output_path: str):
    """绘制各节点内存使用图"""
    if not HAS_MATPLOTLIB or not node_stats:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：峰值内存
    ax1 = axes[0]
    nodes = [s.node_name for s in node_stats]
    peaks = [s.peak_allocated_mb for s in node_stats]

    # 按峰值排序
    sorted_data = sorted(zip(nodes, peaks), key=lambda x: x[1], reverse=True)
    nodes_sorted, peaks_sorted = zip(*sorted_data)

    colors = ['coral' if 'decompose' in n or 'extract' in n or 'decide' in n or 'generate' in n
              else 'steelblue' for n in nodes_sorted]

    bars = ax1.barh(nodes_sorted, peaks_sorted, color=colors, alpha=0.8)

    for bar, peak in zip(bars, peaks_sorted):
        ax1.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                f'{peak:.0f}', va='center', fontsize=9)

    ax1.set_xlabel('Peak Memory (MB)', fontsize=12)
    ax1.set_ylabel('Node', fontsize=12)
    ax1.set_title('Peak GPU Memory by Node\n(Orange = LLM nodes)', fontsize=13)
    ax1.grid(True, alpha=0.3, axis='x')

    # 右图：内存变化量
    ax2 = axes[1]
    deltas = [s.memory_delta_mb for s in node_stats]

    sorted_data = sorted(zip(nodes, deltas), key=lambda x: abs(x[1]), reverse=True)
    nodes_sorted, deltas_sorted = zip(*sorted_data)

    colors = ['green' if d < 0 else 'red' for d in deltas_sorted]
    bars = ax2.barh(nodes_sorted, deltas_sorted, color=colors, alpha=0.8)

    ax2.axvline(x=0, color='black', linewidth=1)
    ax2.set_xlabel('Memory Delta (MB)', fontsize=12)
    ax2.set_ylabel('Node', fontsize=12)
    ax2.set_title('Memory Change per Node\n(Green = Release, Red = Allocate)', fontsize=13)
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved memory by node: {output_path}")


def plot_memory_phases(samples: List[MemorySample], output_path: str):
    """绘制内存阶段分析图（检索 vs LLM）"""
    if not HAS_MATPLOTLIB or not samples:
        return

    # 分类节点
    llm_nodes = {'decompose', 'extract_clues', 'decide', 'generate_final'}
    retrieval_nodes = {'retrieve_hop', 'grade_reranker', 'filter_docs'}

    fig, ax = plt.subplots(figsize=(12, 6))

    times = [s.relative_time_ms for s in samples]
    allocated = [s.allocated_mb for s in samples]

    # 绘制基础曲线
    ax.plot(times, allocated, color='gray', linewidth=1, alpha=0.5)

    # 高亮不同阶段
    llm_times = []
    llm_mem = []
    retrieval_times = []
    retrieval_mem = []
    other_times = []
    other_mem = []

    for s in samples:
        if any(ln in s.current_node for ln in llm_nodes):
            llm_times.append(s.relative_time_ms)
            llm_mem.append(s.allocated_mb)
        elif any(rn in s.current_node for rn in retrieval_nodes):
            retrieval_times.append(s.relative_time_ms)
            retrieval_mem.append(s.allocated_mb)
        else:
            other_times.append(s.relative_time_ms)
            other_mem.append(s.allocated_mb)

    if llm_times:
        ax.scatter(llm_times, llm_mem, color='coral', s=20, alpha=0.6, label='LLM Nodes (GPU Intensive)')
    if retrieval_times:
        ax.scatter(retrieval_times, retrieval_mem, color='steelblue', s=20, alpha=0.6, label='Retrieval Nodes (CPU/IO)')
    if other_times:
        ax.scatter(other_times, other_mem, color='gray', s=10, alpha=0.4, label='Other')

    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Allocated Memory (MB)', fontsize=12)
    ax.set_title('GPU Memory by Execution Phase\n(Retrieval phases show optimization opportunity)', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved memory phases: {output_path}")


def analyze_idle_windows(samples: List[MemorySample]) -> List[Dict[str, Any]]:
    """分析 GPU 空闲窗口"""
    if not samples:
        return []

    llm_nodes = {'decompose', 'extract_clues', 'decide', 'generate_final'}

    windows = []
    window_start = None
    window_samples = []

    for s in samples:
        is_llm = any(ln in s.current_node for ln in llm_nodes)

        if not is_llm and s.current_node:
            # 非 LLM 节点，可能是空闲窗口
            if window_start is None:
                window_start = s.relative_time_ms
            window_samples.append(s)
        else:
            # LLM 节点或无节点，结束当前窗口
            if window_start is not None and window_samples:
                window_end = window_samples[-1].relative_time_ms
                windows.append({
                    "start_ms": window_start,
                    "end_ms": window_end,
                    "duration_ms": window_end - window_start,
                    "avg_memory_mb": sum(s.allocated_mb for s in window_samples) / len(window_samples),
                    "nodes": list(set(s.current_node for s in window_samples)),
                })
            window_start = None
            window_samples = []

    return windows


def main():
    parser = argparse.ArgumentParser(
        description="GPU Memory Timeline Analysis for Hop2Rag"
    )
    parser.add_argument("--limit", type=int, default=3, help="Number of questions to profile")
    parser.add_argument("--k", type=int, default=10, help="Documents per hop")
    parser.add_argument("--max-hops", type=int, default=3, help="Maximum hops")
    parser.add_argument("--sample-interval", type=int, default=10, help="Sample interval in ms")
    parser.add_argument("--skip-plots", action="store_true", help="Skip visualization")

    args = parser.parse_args()

    # 检查环境变量
    if not PERSIST_DIR:
        print("[ERROR] Please set AGRAG_PERSIST_DIR environment variable")
        sys.exit(1)

    print("=" * 70)
    print("GPU Memory Timeline Analysis")
    print("=" * 70)
    print(f"Questions:        {args.limit}")
    print(f"K:                {args.k}")
    print(f"Max Hops:         {args.max_hops}")
    print(f"Sample Interval:  {args.sample_interval} ms")
    print(f"CUDA Available:   {HAS_CUDA}")
    print(f"PERSIST_DIR:      {PERSIST_DIR}")
    print("=" * 70)
    print()

    # 创建输出目录
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 示例问题
    sample_questions = [
        "Who is the director of the movie that won Best Picture at the 2020 Oscars?",
        "What is the capital of the country where the Eiffel Tower is located?",
        "Which university did the founder of Microsoft graduate from?",
    ]
    questions = sample_questions[:args.limit]

    all_results = []

    for i, question in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] Processing: {question[:50]}...")

        # 创建 profiler
        profiler = GPUMemoryProfiler(sample_interval_ms=args.sample_interval)

        # 创建执行器
        executor = InstrumentedHop2Rag(
            profiler=profiler,
            persist_dir=PERSIST_DIR,
            collection_name=COLLECTION_NAME,
            k=args.k,
            max_hops=args.max_hops
        )

        # 开始采样
        profiler.start()

        # 执行
        try:
            result = executor.run(question)
        except Exception as e:
            print(f"  [ERROR] {e}")
            result = {"error": str(e)}

        # 停止采样
        samples = profiler.stop()
        node_stats = profiler.get_node_stats()

        # 分析空闲窗口
        idle_windows = analyze_idle_windows(samples)

        # 记录结果
        request_result = {
            "question": question,
            "num_samples": len(samples),
            "duration_ms": samples[-1].relative_time_ms if samples else 0,
            "peak_memory_mb": max(s.allocated_mb for s in samples) if samples else 0,
            "avg_memory_mb": sum(s.allocated_mb for s in samples) / len(samples) if samples else 0,
            "node_stats": [asdict(s) for s in node_stats],
            "idle_windows": idle_windows,
            "samples": [asdict(s) for s in samples[-100:]],  # 只保存最后100个样本
        }
        all_results.append(request_result)

        print(f"  Samples:     {len(samples)}")
        print(f"  Duration:    {request_result['duration_ms']:.1f} ms")
        print(f"  Peak Memory: {request_result['peak_memory_mb']:.0f} MB")
        print(f"  Idle Windows: {len(idle_windows)}")

        # 为第一个请求绘制详细图
        if i == 0 and not args.skip_plots and HAS_MATPLOTLIB:
            plot_memory_timeline(samples, str(RESULTS_DIR / "memory_timeline.png"))
            plot_memory_by_node(node_stats, str(RESULTS_DIR / "memory_by_node.png"))
            plot_memory_phases(samples, str(RESULTS_DIR / "memory_phases.png"))

    # 保存所有结果
    output_json = RESULTS_DIR / "memory_profile_results.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Saved results: {output_json}")

    # 打印汇总
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    avg_peak = sum(r["peak_memory_mb"] for r in all_results) / len(all_results)
    avg_duration = sum(r["duration_ms"] for r in all_results) / len(all_results)
    total_idle_windows = sum(len(r["idle_windows"]) for r in all_results)
    avg_idle_duration = 0
    if total_idle_windows > 0:
        all_idle = [w for r in all_results for w in r["idle_windows"]]
        avg_idle_duration = sum(w["duration_ms"] for w in all_idle) / len(all_idle)

    print(f"Requests Profiled:      {len(all_results)}")
    print(f"Avg Peak Memory:        {avg_peak:.0f} MB")
    print(f"Avg Duration:           {avg_duration:.0f} ms")
    print(f"Total Idle Windows:     {total_idle_windows}")
    print(f"Avg Idle Window:        {avg_idle_duration:.1f} ms")
    print()

    print("Potential Optimization Opportunities:")
    print("  - Idle windows during retrieval can be used for:")
    print("    * Prefetching next hop documents")
    print("    * KV cache management (offload/load)")
    print("    * Concurrent request scheduling")
    print()
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
