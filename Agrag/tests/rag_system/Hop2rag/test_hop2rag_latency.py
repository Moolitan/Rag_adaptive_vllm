
import argparse
import sys
import os
import json
import time
import threading
import hashlib
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional
import numpy as np
import concurrent.futures

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Try to import seaborn (optional)
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Try to import tokenizer
try:
    from transformers import AutoTokenizer
    HAS_HF_TOKENIZER = True
except ImportError:
    HAS_HF_TOKENIZER = False

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

ROOT = Path(__file__).resolve().parents[3]  # Agrag directory
TESTS_DIR = ROOT / "tests"
RAG_SYSTEM_DIR = TESTS_DIR / "rag_system"
RESULTS_DIR = TESTS_DIR / "results"

HOP2RAG_RESULTS_DIR = RESULTS_DIR / "hop2rag"
INSTRUMENTATION_DIR = HOP2RAG_RESULTS_DIR / "instrumentation"
PLOTS_DIR = HOP2RAG_RESULTS_DIR / "plots"

# Add Agrag to Python path
sys.path.insert(0, str(ROOT))

# 从环境变量获取配置
PERSIST_DIR = os.environ.get("AGRAG_PERSIST_DIR", "")
COLLECTION_NAME = os.environ.get("AGRAG_COLLECTION_NAME", "hotpot_fullwiki")

# 导入新的独立 Rag 模块
from Rag.hop2_rag import (
    run_hop2_rag,
    get_hop2_rag_app,
    build_hop2_rag,
    enable_instrumentation,
    disable_instrumentation,
    clear_instrumentation_log,
    save_workflow_graph,
)
from tests.cores import load_hotpotqa_fullwiki

# Publication-quality plot settings
if HAS_MATPLOTLIB:
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.figsize': (8, 5),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'lines.linewidth': 2,
        'lines.markersize': 8,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


# =============================
# Tokenizer Utilities
# =============================

class TokenCounter:
    """Token counter with multiple backend support"""

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.tokenizer = None
        self.backend = "unknown"

        # Try HuggingFace tokenizer first
        if HAS_HF_TOKENIZER:
            try:
                # Try to load a common tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "Qwen/Qwen2.5-7B-Instruct",
                    trust_remote_code=True
                )
                self.backend = "huggingface"
            except Exception:
                pass

        # Fallback to tiktoken
        if self.tokenizer is None and HAS_TIKTOKEN:
            try:
                self.tokenizer = tiktoken.encoding_for_model(model_name)
                self.backend = "tiktoken"
            except Exception:
                try:
                    self.tokenizer = tiktoken.get_encoding("cl100k_base")
                    self.backend = "tiktoken"
                except Exception:
                    pass

    def count(self, text: str) -> int:
        """Count tokens in text"""
        if not text:
            return 0

        if self.tokenizer is None:
            return 0

        try:
            if self.backend == "huggingface":
                return len(self.tokenizer.encode(text))
            elif self.backend == "tiktoken":
                return len(self.tokenizer.encode(text))
        except Exception:
            return 0

        return 0

    def get_source(self) -> str:
        """Get token source identifier"""
        if self.backend == "huggingface":
            return "tokenizer_hf"
        elif self.backend == "tiktoken":
            return "tokenizer_tiktoken"
        return "unknown"


# Global token counter (lazy init)
_token_counter: Optional[TokenCounter] = None

def get_token_counter() -> TokenCounter:
    global _token_counter
    if _token_counter is None:
        _token_counter = TokenCounter()
    return _token_counter


def estimate_prompt_tokens(question: str, documents: List[Any]) -> int:
    """
    Estimate prompt tokens by reconstructing approximate prompt structure.
    This mimics the actual prompt construction in hop2_rag.py.
    """
    counter = get_token_counter()

    # Construct approximate prompt (matching MULTI_HOP_RAG_PROMPT structure)
    context_parts = []
    for i, doc in enumerate(documents):
        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        context_parts.append(f"[Doc {i+1}] {content}")

    context = "\n\n".join(context_parts)

    # Approximate prompt template
    prompt = f"""Answer a multi-hop question using evidence from multiple retrieval steps.

Original question: {question}

=== Evidence from multiple hops ===
{context}

Hop history:
[hop history placeholder]

Instructions:
1. Synthesize information across ALL hops
2. For comparison questions: Compare attributes found in different hops
3. For bridge questions: Connect intermediate entity to final answer
4. Ground your answer in the evidence provided
5. Give a direct, concise answer

Answer:"""

    return counter.count(prompt)


@dataclass
class RequestTrace:
    """Request-level trace for system analysis"""
    request_id: str
    question: str

    # Timing (seconds)
    submit_time: float
    start_time: float
    end_time: float

    # Latency breakdown
    e2e_latency: float          # End-to-end latency
    ttft: float = 0.0           # Time to first token (if measurable)
    retrieval_time: float = 0.0 # Retrieval stage time
    llm_time: float = 0.0       # LLM inference time

    # Token statistics (A: proper token counting)
    prompt_tokens: int = 0      # Tokens in prompt sent to LLM
    completion_tokens: int = 0  # Generated tokens
    total_tokens: int = 0       # prompt + completion
    token_source: str = "unknown"  # {"usage", "tokenizer_estimate", "unknown"}

    # Legacy fields (kept for backward compatibility)
    context_tokens: int = 0     # Alias for prompt_tokens
    output_tokens: int = 0      # Alias for completion_tokens

    # Multi-hop specific
    num_hops: int = 0           # Number of hops executed
    hop_queries: List[str] = None
    hop_docs_per_hop: List[int] = None

    # System state
    concurrent_requests: int = 0 # Concurrent requests at start

    # Result
    success: bool = True
    error: str = ""

    def __post_init__(self):
        if self.hop_queries is None:
            self.hop_queries = []
        if self.hop_docs_per_hop is None:
            self.hop_docs_per_hop = []


class SystemBenchmark:
    """System-level benchmark runner with request-level tracing"""

    def __init__(self, persist_dir: str, collection_name: str, retrieval_k: int, max_hops: int = 5):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.retrieval_k = retrieval_k
        self.max_hops = max_hops
        self.app = get_hop2_rag_app()
        self.token_counter = get_token_counter()

        # Shared state for concurrency tracking
        self.active_requests = 0
        self.lock = threading.Lock()

    def run_single_request(self, req_id: str, question: str, submit_time: float) -> RequestTrace:
        """Run a single request and collect trace"""

        with self.lock:
            self.active_requests += 1
            concurrent = self.active_requests

        start_time = time.time()

        trace = RequestTrace(
            request_id=req_id,
            question=question,
            submit_time=submit_time,
            start_time=start_time,
            end_time=0.0,
            e2e_latency=0.0,
            concurrent_requests=concurrent
        )

        try:
            # 使用新的独立 run_hop2_rag 接口
            result = run_hop2_rag(
                question=question,
                persist_dir=self.persist_dir,
                collection_name=self.collection_name,
                k=self.retrieval_k,
                max_hops=self.max_hops
            )

            end_time = time.time()
            metadata = result.get("metadata", {})

            # Extract multi-hop info
            trace.num_hops = metadata.get("total_hops", 0)
            trace.hop_queries = metadata.get("hop_queries", [])

            # [A] Token counting - proper implementation
            all_docs = result.get("documents", [])
            answer = result.get("answer", "")

            # Check if result contains LLM usage stats
            usage = result.get("usage", {})
            if usage and "prompt_tokens" in usage:
                # Use actual usage from LLM
                trace.prompt_tokens = usage.get("prompt_tokens", 0)
                trace.completion_tokens = usage.get("completion_tokens", 0)
                trace.total_tokens = usage.get("total_tokens", 0)
                trace.token_source = "usage"
            else:
                # Estimate using tokenizer
                trace.prompt_tokens = estimate_prompt_tokens(question, all_docs)
                trace.completion_tokens = self.token_counter.count(answer)
                trace.total_tokens = trace.prompt_tokens + trace.completion_tokens
                trace.token_source = self.token_counter.get_source()

            # Backward compatibility
            trace.context_tokens = trace.prompt_tokens
            trace.output_tokens = trace.completion_tokens

            trace.end_time = end_time
            trace.e2e_latency = end_time - start_time
            trace.success = True

        except Exception as e:
            trace.end_time = time.time()
            trace.e2e_latency = trace.end_time - start_time
            trace.success = False
            trace.error = str(e)

        finally:
            with self.lock:
                self.active_requests -= 1

        return trace

    def run_serial(self, questions: List[tuple], verbose: bool = True) -> List[RequestTrace]:
        """Run requests serially (concurrency=1)"""
        traces = []

        for i, (req_id, question) in enumerate(questions):
            if verbose:
                print(f"[{i+1}/{len(questions)}] Request {req_id}...")

            submit_time = time.time()
            trace = self.run_single_request(req_id, question, submit_time)
            traces.append(trace)

            if verbose:
                print(f"  ✓ Latency: {trace.e2e_latency:.2f}s, Hops: {trace.num_hops}, Tokens: {trace.prompt_tokens} ({trace.token_source})")

        return traces

    def run_concurrent(self, questions: List[tuple], max_workers: int, verbose: bool = True) -> List[RequestTrace]:
        """Run requests with controlled concurrency"""
        traces = []

        # [C] Each request gets its own submit_time
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for req_id, question in questions:
                submit_time = time.time()  # Individual submit time
                future = executor.submit(self.run_single_request, req_id, question, submit_time)
                futures.append(future)

            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                if verbose:
                    print(f"[{i+1}/{len(questions)}] Request completed")
                trace = future.result()
                traces.append(trace)

        return traces


def compute_statistics(traces: List[RequestTrace]) -> Dict[str, Any]:
    """Compute system-level statistics from traces"""

    if not traces:
        return {}

    successful_traces = [t for t in traces if t.success]
    latencies = [t.e2e_latency for t in successful_traces]
    n = len(latencies)

    if n == 0:
        return {"error": "No successful requests"}

    # [C] Percentiles using np.percentile (statistically correct)
    latencies_arr = np.array(latencies)
    p50 = float(np.percentile(latencies_arr, 50))
    p95 = float(np.percentile(latencies_arr, 95))
    p99 = float(np.percentile(latencies_arr, 99))

    # [C] Timing with proper wall time calculation
    earliest_submit = min(t.submit_time for t in traces)
    latest_end = max(t.end_time for t in traces)
    wall_time = latest_end - earliest_submit
    throughput = len(successful_traces) / wall_time if wall_time > 0 else 0

    # Queue time statistics (for concurrent runs)
    queue_times = [t.start_time - t.submit_time for t in successful_traces]
    avg_queue_time = np.mean(queue_times) if queue_times else 0
    p95_queue_time = float(np.percentile(queue_times, 95)) if len(queue_times) > 0 else 0

    # Token statistics
    prompt_tokens = [t.prompt_tokens for t in successful_traces]
    completion_tokens = [t.completion_tokens for t in successful_traces]
    total_tokens = [t.total_tokens for t in successful_traces]

    avg_prompt_tokens = np.mean(prompt_tokens) if prompt_tokens else 0
    avg_completion_tokens = np.mean(completion_tokens) if completion_tokens else 0
    avg_total_tokens = np.mean(total_tokens) if total_tokens else 0

    # Token source (should be consistent)
    token_sources = list(set(t.token_source for t in successful_traces))
    token_source = token_sources[0] if len(token_sources) == 1 else "mixed"

    # Hop statistics
    hops = [t.num_hops for t in successful_traces]
    avg_hops = np.mean(hops) if hops else 0

    # Hop distribution
    hop_counts = {}
    for t in successful_traces:
        h = t.num_hops
        hop_counts[h] = hop_counts.get(h, 0) + 1

    return {
        "total_requests": len(traces),
        "successful_requests": len(successful_traces),
        "failed_requests": len(traces) - len(successful_traces),

        # Latency (seconds)
        "latency_mean": float(np.mean(latencies_arr)),
        "latency_std": float(np.std(latencies_arr)),
        "latency_median": p50,
        "latency_p95": p95,
        "latency_p99": p99,
        "latency_min": float(np.min(latencies_arr)),
        "latency_max": float(np.max(latencies_arr)),

        # [C] Throughput (corrected)
        "wall_time_sec": wall_time,
        "throughput_req_per_sec": throughput,
        "avg_queue_time_sec": float(avg_queue_time),
        "p95_queue_time_sec": p95_queue_time,

        # [A] Token statistics
        "avg_prompt_tokens": float(avg_prompt_tokens),
        "avg_completion_tokens": float(avg_completion_tokens),
        "avg_total_tokens": float(avg_total_tokens),
        "token_source": token_source,

        # Legacy (backward compatibility)
        "avg_context_tokens": float(avg_prompt_tokens),
        "avg_output_tokens": float(avg_completion_tokens),

        # Multi-hop
        "avg_hops": float(avg_hops),
        "hop_distribution": hop_counts,
    }


def save_traces(traces: List[RequestTrace], output_path: str):
    """Save traces to JSON file"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([asdict(t) for t in traces], f, ensure_ascii=False, indent=2)


def load_traces(trace_file: str) -> List[Dict[str, Any]]:
    """Load request traces from JSON file"""
    with open(trace_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_jsonl(filepath: str):
    """Load JSONL file"""
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def plot_latency_cdf(traces_dict: Dict[str, List[Dict]], output_file: str):
    """Plot Cumulative Distribution Function (CDF) of end-to-end latency"""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for workflow_name, traces in traces_dict.items():
        latencies = sorted([t['e2e_latency'] for t in traces if t['success']])
        n = len(latencies)
        if n == 0:
            continue

        cdf = np.arange(1, n + 1) / n
        ax.plot(latencies, cdf, label=workflow_name, linewidth=2.5, marker='o', markersize=4, markevery=max(1, n//10))

    ax.set_xlabel('End-to-End Latency (seconds)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Latency CDF: Tail Behavior Analysis')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Mark P95 and P99 lines
    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=0.99, color='darkred', linestyle='--', alpha=0.5, linewidth=1)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def plot_context_vs_latency(traces: List[Dict], output_file: str):
    """Scatter plot: prompt tokens vs latency (renamed from context_tokens)"""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # [A] Use prompt_tokens instead of context_tokens
    prompt_tokens = [t.get('prompt_tokens', t.get('context_tokens', 0)) for t in traces if t['success']]
    latencies = [t['e2e_latency'] for t in traces if t['success']]
    hops = [t['num_hops'] for t in traces if t['success']]

    scatter = ax.scatter(prompt_tokens, latencies, c=hops, cmap='viridis',
                        s=100, alpha=0.6, edgecolors='black', linewidths=0.5)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Number of Hops')

    ax.set_xlabel('Prompt Tokens')
    ax.set_ylabel('End-to-End Latency (seconds)')
    ax.set_title('Prompt Tokens vs Latency (colored by hops)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def plot_hop_distribution(traces: List[Dict], output_file: str):
    """Bar chart: hop count distribution"""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    hop_counts = {}
    for t in traces:
        h = t['num_hops']
        hop_counts[h] = hop_counts.get(h, 0) + 1

    hops = sorted(hop_counts.keys())
    counts = [hop_counts[h] for h in hops]

    # Use default colors if seaborn not available
    if HAS_SEABORN:
        color = sns.color_palette("Set2")[0]
    else:
        color = '#66c2a5'

    bars = ax.bar(hops, counts, color=color, alpha=0.8, edgecolor='black')

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom', fontsize=11)

    ax.set_xlabel('Number of Hops')
    ax.set_ylabel('Number of Requests')
    ax.set_title('Hop Distribution')
    ax.set_xticks(hops)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def plot_latency_by_hops(traces: List[Dict], output_file: str):
    """Boxplot: latency grouped by number of hops"""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    hop_latencies = {}
    for t in traces:
        if t['success']:
            h = t['num_hops']
            if h not in hop_latencies:
                hop_latencies[h] = []
            hop_latencies[h].append(t['e2e_latency'])

    if not hop_latencies:
        plt.close()
        return

    hops = sorted(hop_latencies.keys())
    data = [hop_latencies[h] for h in hops]
    labels = [f'{h} hops' for h in hops]

    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)

    # Use default colors if seaborn not available
    if HAS_SEABORN:
        colors = sns.color_palette("coolwarm", len(data))
    else:
        colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(data)))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('End-to-End Latency (seconds)')
    ax.set_title('Latency vs Number of Hops')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def plot_tokens_histogram(traces: List[Dict], output_file: str):
    """[D] Histogram of token counts"""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    prompt_tokens = [t.get('prompt_tokens', 0) for t in traces if t['success']]
    completion_tokens = [t.get('completion_tokens', 0) for t in traces if t['success']]

    # Prompt tokens histogram
    if prompt_tokens and max(prompt_tokens) > 0:
        axes[0].hist(prompt_tokens, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Prompt Tokens')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Prompt Token Distribution')
        axes[0].axvline(np.mean(prompt_tokens), color='red', linestyle='--', label=f'Mean: {np.mean(prompt_tokens):.0f}')
        axes[0].legend()

    # Completion tokens histogram
    if completion_tokens and max(completion_tokens) > 0:
        axes[1].hist(completion_tokens, bins=20, color='forestgreen', alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Completion Tokens')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Completion Token Distribution')
        axes[1].axvline(np.mean(completion_tokens), color='red', linestyle='--', label=f'Mean: {np.mean(completion_tokens):.0f}')
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def plot_latency_vs_prompt_tokens(traces: List[Dict], output_file: str):
    """[D] Scatter plot: latency vs prompt tokens with regression line"""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    prompt_tokens = np.array([t.get('prompt_tokens', 0) for t in traces if t['success']])
    latencies = np.array([t['e2e_latency'] for t in traces if t['success']])

    if len(prompt_tokens) == 0 or max(prompt_tokens) == 0:
        plt.close()
        return

    ax.scatter(prompt_tokens, latencies, alpha=0.6, s=80, edgecolors='black', linewidths=0.5)

    # Add trend line
    if len(prompt_tokens) > 2:
        z = np.polyfit(prompt_tokens, latencies, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(prompt_tokens), max(prompt_tokens), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=f'Trend: {z[0]:.4f}x + {z[1]:.2f}')
        ax.legend()

    ax.set_xlabel('Prompt Tokens')
    ax.set_ylabel('End-to-End Latency (seconds)')
    ax.set_title('Latency vs Prompt Tokens')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def plot_node_latency_breakdown(node_times, output_path):
    """Plot node latency breakdown as horizontal bar chart"""
    if not HAS_MATPLOTLIB:
        return

    node_avg = [(node, np.mean(times)) for node, times in node_times.items()]
    node_avg.sort(key=lambda x: x[1], reverse=True)

    nodes, avg_times = zip(*node_avg)

    plt.figure(figsize=(10, 6))
    plt.barh(nodes, avg_times, color='steelblue')
    plt.xlabel('Average Latency (ms)', fontsize=12)
    plt.ylabel('Node', fontsize=12)
    plt.title('Per-Node Latency Breakdown (Hop2Rag)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_instrumentation_latency_cdf(total_latencies, output_path):
    """Plot CDF of total latencies from instrumentation"""
    if not HAS_MATPLOTLIB:
        return

    sorted_latencies = np.sort(total_latencies)
    cdf = np.arange(1, len(sorted_latencies) + 1) / len(sorted_latencies)

    plt.figure(figsize=(8, 5))
    plt.plot(sorted_latencies, cdf, linewidth=2, color='darkblue')
    plt.xlabel('Total Latency (ms)', fontsize=12)
    plt.ylabel('CDF', fontsize=12)
    plt.title('Request Latency Distribution (CDF)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_latency_pie_chart(node_times, edge_times, output_path):
    """Plot pie chart showing node and edge latency breakdown (absolute values)"""
    if not HAS_MATPLOTLIB:
        return

    # 计算各节点/边的平均耗时
    node_avg = {node: np.mean(times) for node, times in node_times.items()}
    edge_avg = {edge: np.mean(times) for edge, times in edge_times.items()}

    # 合并所有组件
    all_components = {}
    for node, avg in node_avg.items():
        all_components[f"[Node] {node}"] = avg
    for edge, avg in edge_avg.items():
        all_components[f"[Edge] {edge}"] = avg

    # 按耗时排序
    sorted_components = sorted(all_components.items(), key=lambda x: x[1], reverse=True)

    # 合并小于总耗时2%的组件为"其他"
    total_time = sum(v for _, v in sorted_components)
    if total_time == 0:
        return

    threshold = total_time * 0.02

    main_components = []
    other_time = 0
    for name, time_ms in sorted_components:
        if time_ms >= threshold:
            main_components.append((name, time_ms))
        else:
            other_time += time_ms

    if other_time > 0:
        main_components.append(("Others", other_time))

    labels = [name for name, _ in main_components]
    sizes = [time_ms for _, time_ms in main_components]

    # 创建饼状图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # 左图：饼状图
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    explode = [0.02] * len(labels)  # 轻微分离每个扇形

    wedges, texts, autotexts = ax1.pie(
        sizes,
        explode=explode,
        labels=None,  # 不在饼图上显示标签
        colors=colors,
        autopct=lambda pct: f'{pct:.1f}%' if pct > 3 else '',
        shadow=False,
        startangle=90,
        pctdistance=0.75
    )

    # 设置百分比文字样式
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')

    ax1.set_title('Latency Distribution by Component', fontsize=14, fontweight='bold')

    # 右图：带绝对值的图例
    legend_labels = [f"{name}\n{time_ms:.1f} ms ({time_ms/total_time*100:.1f}%)"
                     for name, time_ms in main_components]

    ax2.axis('off')
    ax2.legend(
        wedges,
        legend_labels,
        title=f"Total: {total_time:.1f} ms",
        loc='center',
        fontsize=10,
        title_fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=True
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_node_latency_stacked_bar(node_times, output_path):
    """Plot stacked bar chart showing node latency breakdown per request"""
    if not HAS_MATPLOTLIB:
        return

    # 获取所有节点名
    all_nodes = list(node_times.keys())
    n_requests = len(list(node_times.values())[0]) if node_times else 0

    if n_requests == 0:
        return

    # 创建堆叠条形图
    fig, ax = plt.subplots(figsize=(12, 6))

    # 准备数据
    bottom = np.zeros(n_requests)
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_nodes)))

    for i, node in enumerate(all_nodes):
        times = node_times[node]
        ax.bar(range(n_requests), times, bottom=bottom, label=node, color=colors[i], alpha=0.8)
        bottom += np.array(times)

    ax.set_xlabel('Request Index', fontsize=12)
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('Per-Request Node Latency Breakdown', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_system_benchmark(examples, k, max_hops, concurrency, verbose):
    """Run system-level benchmark (latency, throughput, tail behavior)"""
    print("[Step 1/3] Running system-level benchmark...")
    trace_file = HOP2RAG_RESULTS_DIR / "hop2rag_traces.json"

    # Prepare questions
    questions = [(f"req_{i}", ex.question) for i, ex in enumerate(examples)]

    # Run benchmark
    bench = SystemBenchmark(
        persist_dir=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
        retrieval_k=k,
        max_hops=max_hops
    )

    try:
        if concurrency == 1:
            traces = bench.run_serial(questions, verbose=verbose)
        else:
            traces = bench.run_concurrent(questions, max_workers=concurrency, verbose=verbose)

        # Compute statistics
        stats = compute_statistics(traces)

        # Save traces
        save_traces(traces, trace_file)

        # Save statistics
        stats_file = Path(trace_file).with_suffix('.stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        print()
        return trace_file

    finally:
        pass  # No cleanup needed for standalone app


def run_instrumentation(examples, k, max_hops, verbose):
    """Run with detailed node/edge instrumentation"""
    print("[Step 2/3] Running instrumentation benchmark...")
    print("  (Collecting node/edge timing data for ablation studies)")

    # 确保目录存在
    INSTRUMENTATION_DIR.mkdir(parents=True, exist_ok=True)

    instrumentation_log = INSTRUMENTATION_DIR / "rag_timings.jsonl"

    # 启用插桩，直接输出到目标目录
    enable_instrumentation(str(INSTRUMENTATION_DIR), "rag_timings.jsonl")

    # 清空旧日志
    clear_instrumentation_log()

    # Run each query with instrumentation
    for i, item in enumerate(examples):
        question = item.question

        try:
            result = run_hop2_rag(
                question=question,
                persist_dir=PERSIST_DIR,
                collection_name=COLLECTION_NAME,
                k=k,
                max_hops=max_hops
            )

            if verbose:
                hops = result.get("metadata", {}).get("total_hops", 0)
                print(f"  [{i+1}/{len(examples)}] Hops: {hops}, Time: {result.get('timing_ms', 0):.2f} ms")

        except Exception as e:
            print(f"  [{i+1}/{len(examples)}] Query failed: {e}")

    # 禁用插桩
    disable_instrumentation()

    print(f"✓ Instrumentation log saved: {instrumentation_log}")
    print()

    return instrumentation_log


def analyze_instrumentation(instrumentation_log):
    """Analyze instrumentation data and print summary"""
    if not instrumentation_log.exists():
        return

    print("=" * 80)
    print("INSTRUMENTATION SUMMARY")
    print("=" * 80)
    print()

    # Load all records
    records = load_jsonl(instrumentation_log)

    if not records:
        print("No instrumentation data found.")
        return

    # Total latency stats
    total_latencies = [r['total_ms'] for r in records]
    print(f"Total Latency (ms) - {len(records)} requests:")
    print(f"  Mean:   {np.mean(total_latencies):8.2f} ms")
    print(f"  Median: {np.median(total_latencies):8.2f} ms")
    print(f"  P95:    {np.percentile(total_latencies, 95):8.2f} ms")
    print(f"  P99:    {np.percentile(total_latencies, 99):8.2f} ms")
    print()

    # Node timing stats
    print("Node Timings (ms) - Average across all requests:")
    node_times = defaultdict(list)
    for r in records:
        for node, ms in r.get('node_ms', {}).items():
            node_times[node].append(ms)

    if node_times:
        node_avg = [(node, np.mean(times)) for node, times in node_times.items()]
        node_avg.sort(key=lambda x: x[1], reverse=True)

        total_avg = sum(avg for _, avg in node_avg)
        for node, avg in node_avg:
            pct = (avg / total_avg * 100) if total_avg > 0 else 0
            p95 = np.percentile(node_times[node], 95)
            print(f"  {node:45s} avg={avg:7.2f} ms  p95={p95:7.2f} ms  ({pct:5.1f}%)")
        print(f"  {'TOTAL':45s} {total_avg:7.2f} ms")
    print()

    # Edge timing stats
    print("Edge Timings (ms) - Average across all requests:")
    edge_times = defaultdict(list)
    for r in records:
        for edge, ms in r.get('edge_ms', {}).items():
            edge_times[edge].append(ms)

    if edge_times:
        for edge, times in sorted(edge_times.items()):
            avg = np.mean(times)
            print(f"  {edge:45s} {avg:7.2f} ms")
    print()

    return {
        'total_latencies': total_latencies,
        'node_times': node_times,
        'edge_times': edge_times
    }


def generate_visualizations(trace_file, instrumentation_log, skip_plots):
    """Generate visualization plots"""
    if skip_plots:
        print("[Step 3/3] Skipping visualization (--skip-plots)")
        return

    print("[Step 3/3] Generating visualization plots...")

    # 生成工作流图
    workflow_graph_path = str(PLOTS_DIR / 'hop2rag_workflow.png')
    if save_workflow_graph(workflow_graph_path):
        print(f"  ✓ Workflow graph saved: {workflow_graph_path}")
    else:
        print("  ⚠ Could not generate workflow graph (missing dependencies)")

    if not HAS_MATPLOTLIB:
        print("  Matplotlib not available, skipping plots")
        return

    # System-level plots
    traces = load_traces(trace_file)
    workflow_name = Path(trace_file).stem.replace('_traces', '')

    plot_latency_cdf(
        {workflow_name: traces},
        str(PLOTS_DIR / 'latency_cdf.png')
    )

    plot_context_vs_latency(
        traces,
        str(PLOTS_DIR / 'prompt_tokens_vs_latency.png')  # Renamed
    )

    plot_hop_distribution(
        traces,
        str(PLOTS_DIR / 'hop_distribution.png')
    )

    plot_latency_by_hops(
        traces,
        str(PLOTS_DIR / 'latency_by_hops.png')
    )

    # [D] New plots
    plot_tokens_histogram(
        traces,
        str(PLOTS_DIR / 'tokens_histogram.png')
    )

    plot_latency_vs_prompt_tokens(
        traces,
        str(PLOTS_DIR / 'latency_vs_prompt_tokens.png')
    )

    # Instrumentation plots
    if instrumentation_log and instrumentation_log.exists():
        records = load_jsonl(instrumentation_log)
        if records:
            # Total latency stats
            total_latencies = [r['total_ms'] for r in records]

            # Node timing stats
            node_times = defaultdict(list)
            for r in records:
                for node, ms in r.get('node_ms', {}).items():
                    node_times[node].append(ms)

            # Edge timing stats
            edge_times = defaultdict(list)
            for r in records:
                for edge, ms in r.get('edge_ms', {}).items():
                    edge_times[edge].append(ms)

            if node_times:
                plot_node_latency_breakdown(
                    node_times,
                    str(PLOTS_DIR / 'node_latency_breakdown.png')
                )

            if total_latencies:
                plot_instrumentation_latency_cdf(
                    total_latencies,
                    str(PLOTS_DIR / 'instrumentation_latency_cdf.png')
                )

            # 饼状图：节点和边的延迟占比
            if node_times or edge_times:
                plot_latency_pie_chart(
                    node_times,
                    edge_times,
                    str(PLOTS_DIR / 'latency_pie_chart.png')
                )

            # 堆叠条形图：每个请求的节点延迟分解
            if node_times and len(list(node_times.values())[0]) > 1:
                plot_node_latency_stacked_bar(
                    node_times,
                    str(PLOTS_DIR / 'node_latency_stacked_bar.png')
                )

    print()


def print_summary(trace_file, instrumentation_log, skip_plots):
    """Print final summary"""
    print("=" * 80)
    print("TEST COMPLETE!")
    print("=" * 80)
    print()
    print("Results saved to:")
    print(f"  System traces:       {trace_file}")
    print(f"  System stats:        {trace_file.with_suffix('.stats.json')}")

    if instrumentation_log and instrumentation_log.exists():
        print(f"  Instrumentation:     {instrumentation_log}")

    print(f"  Plots:               {PLOTS_DIR}/")
    print()

    # Print hop distribution summary
    stats_file = trace_file.with_suffix('.stats.json')
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        hop_dist = stats.get("hop_distribution", {})
        print("Hop Distribution:")
        for h in sorted(hop_dist.keys(), key=lambda x: int(x)):
            print(f"  {h} hops: {hop_dist[h]} requests")
        print(f"  Average hops: {stats.get('avg_hops', 0):.2f}")
        print()

        print("Token Statistics:")
        print(f"  Avg prompt tokens:     {stats.get('avg_prompt_tokens', 0):.0f}")
        print(f"  Avg completion tokens: {stats.get('avg_completion_tokens', 0):.0f}")
        print(f"  Token source:          {stats.get('token_source', 'unknown')}")
        print()

    if not skip_plots and HAS_MATPLOTLIB:
        print("Generated plots:")
        print("  - latency_cdf.png                (Tail latency distribution)")
        print("  - prompt_tokens_vs_latency.png   (Token growth analysis)")
        print("  - hop_distribution.png           (Hop count histogram)")
        print("  - latency_by_hops.png            (Latency vs hops)")
        print("  - tokens_histogram.png           (Token distribution)")
        print("  - latency_vs_prompt_tokens.png   (Latency correlation)")
        if instrumentation_log and instrumentation_log.exists():
            print("  - node_latency_breakdown.png     (Node timing breakdown)")
            print("  - instrumentation_latency_cdf.png (Instrumentation CDF)")
            print("  - latency_pie_chart.png          (Component breakdown)")
            print("  - node_latency_stacked_bar.png   (Per-request breakdown)")
        print()


def main():
    ap = argparse.ArgumentParser(
        description="Test Hop2Rag multi-hop implementation with performance optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic system benchmark
  python test_hop2rag_latency.py --limit 10 --k 20

  # With detailed instrumentation
  python test_hop2rag_latency.py --limit 10 --k 20 --enable-instrumentation

  # High concurrency test
  python test_hop2rag_latency.py --limit 100 --k 20 --concurrency 10
        """
    )

    ap.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of questions to test (default: 10)"
    )
    ap.add_argument(
        "--k",
        type=int,
        default=20,
        help="Documents to retrieve per hop (default: 20)"
    )
    ap.add_argument(
        "--max-hops",
        type=int,
        default=5,
        help="Maximum number of hops (default: 5)"
    )
    ap.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Concurrent requests (default: 1)"
    )
    ap.add_argument(
        "--enable-instrumentation",
        action="store_true",
        help="Enable detailed node/edge timing instrumentation"
    )
    ap.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip visualization generation"
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = ap.parse_args()

    # 检查环境变量
    if not PERSIST_DIR:
        print("[ERROR] 请设置环境变量 AGRAG_PERSIST_DIR")
        sys.exit(1)

    print("=" * 80)
    print("Hop2Rag Performance Benchmark Suite")
    print("=" * 80)
    print(f"Questions:          {args.limit}")
    print(f"Retrieval K:        {args.k}")
    print(f"Max Hops:           {args.max_hops}")
    print(f"Concurrency:        {args.concurrency}")
    print(f"Instrumentation:    {'Enabled' if args.enable_instrumentation else 'Disabled'}")
    print(f"Token Counter:      {get_token_counter().backend}")
    print(f"Results:            {HOP2RAG_RESULTS_DIR}")
    print(f"PERSIST_DIR:        {PERSIST_DIR}")
    print(f"COLLECTION_NAME:    {COLLECTION_NAME}")
    print("=" * 80)
    print()

    # Create directories
    HOP2RAG_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.enable_instrumentation:
        INSTRUMENTATION_DIR.mkdir(parents=True, exist_ok=True)

    # Load data once
    print("Loading HotpotQA dataset...")
    hotpot_path = ROOT / "data" / "hotpotqa" / "hotpot_dev_fullwiki_v1.json"
    examples = load_hotpotqa_fullwiki(
        path=str(hotpot_path),
        limit=args.limit,
        persist_dir=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
        retrieval_k=args.k,
    )
    print(f"Loaded {len(examples)} questions\n")

    # Step 1: System benchmark
    trace_file = run_system_benchmark(
        examples, args.k, args.max_hops,
        args.concurrency, args.verbose
    )

    # Step 2: Instrumentation
    instrumentation_log = None
    if args.enable_instrumentation:
        instrumentation_log = run_instrumentation(
            examples, args.k, args.max_hops, args.verbose
        )
        stats = analyze_instrumentation(instrumentation_log)

    # Step 3: Visualization
    generate_visualizations(trace_file, instrumentation_log, args.skip_plots)

    # Summary
    print_summary(trace_file, instrumentation_log, args.skip_plots)


if __name__ == "__main__":
    main()
