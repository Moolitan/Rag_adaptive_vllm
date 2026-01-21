#!/usr/bin/env python3
"""
实验一：KV Cache 复用潜力分析

目标：分析 Agentic RAG 中多次 LLM 调用之间的 prompt prefix 重叠程度，
     评估 KV Cache 复用的理论上限。

观察点：
- 同一请求内，不同节点（decompose, extract_clues, decide, generate）的 prompt 重叠
- 同一请求内，不同跳数之间的 prompt 重叠（累积上下文）
- 跨请求的 prompt 重叠（System Prompt, Instruction 等固定部分）

输出：
- results/hop2rag/kv_cache/prefix_overlap_analysis.json
- results/hop2rag/kv_cache/prefix_overlap_heatmap.png
- results/hop2rag/kv_cache/token_breakdown.png
"""

import argparse
import json
import sys
import os
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

# Setup paths
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# Try to import tokenizer
try:
    from transformers import AutoTokenizer
    HAS_TOKENIZER = True
except ImportError:
    HAS_TOKENIZER = False
    print("[WARN] transformers not found, using simple word-based tokenization")

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

# Import prompts from hop2_rag
from Rag.hop2_rag import (
    QUESTION_DECOMPOSER_PROMPT,
    CLUE_EXTRACTOR_PROMPT,
    HOP_DECISION_PROMPT,
    MULTI_HOP_RAG_PROMPT,
)

# Results directory
RESULTS_DIR = ROOT / "tests" / "results" / "hop2rag" / "kv_cache"


@dataclass
class LLMCall:
    """记录单次 LLM 调用的信息"""
    request_id: str
    node_name: str          # decompose, extract_clues, decide, generate
    hop: int
    prompt: str
    prompt_tokens: List[str]
    token_count: int
    timestamp: float

    # Prefix 分析
    prefix_hashes: List[str] = field(default_factory=list)  # 每 N tokens 的 hash


@dataclass
class PrefixOverlapResult:
    """Prefix 重叠分析结果"""
    call_a: str  # node_name@hop
    call_b: str
    shared_prefix_tokens: int
    total_tokens_a: int
    total_tokens_b: int
    overlap_ratio: float  # shared / min(a, b)


class SimpleTokenizer:
    """简单的基于词的分词器（备用）"""

    def encode(self, text: str) -> List[int]:
        words = text.split()
        return list(range(len(words)))

    def tokenize(self, text: str) -> List[str]:
        return text.split()


class KVCacheAnalyzer:
    """KV Cache 复用潜力分析器"""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.model_name = model_name

        if HAS_TOKENIZER:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                print(f"[INFO] Loaded tokenizer: {model_name}")
            except Exception as e:
                print(f"[WARN] Failed to load tokenizer {model_name}: {e}")
                print("[WARN] Falling back to simple tokenizer")
                self.tokenizer = SimpleTokenizer()
        else:
            self.tokenizer = SimpleTokenizer()

        self.calls: List[LLMCall] = []
        self.hash_granularity = 64  # 每 64 tokens 计算一个 hash

    def tokenize(self, text: str) -> List[str]:
        """分词"""
        if hasattr(self.tokenizer, 'tokenize'):
            return self.tokenizer.tokenize(text)
        else:
            return text.split()

    def compute_prefix_hashes(self, tokens: List[str]) -> List[str]:
        """计算 prefix hashes，每 N tokens 一个"""
        hashes = []
        for i in range(self.hash_granularity, len(tokens) + 1, self.hash_granularity):
            prefix = " ".join(tokens[:i])
            h = hashlib.md5(prefix.encode()).hexdigest()[:8]
            hashes.append(h)
        return hashes

    def record_call(
        self,
        request_id: str,
        node_name: str,
        hop: int,
        prompt: str
    ) -> LLMCall:
        """记录一次 LLM 调用"""
        tokens = self.tokenize(prompt)
        prefix_hashes = self.compute_prefix_hashes(tokens)

        call = LLMCall(
            request_id=request_id,
            node_name=node_name,
            hop=hop,
            prompt=prompt,
            prompt_tokens=tokens,
            token_count=len(tokens),
            timestamp=time.time(),
            prefix_hashes=prefix_hashes
        )
        self.calls.append(call)
        return call

    def compute_prefix_overlap(self, call_a: LLMCall, call_b: LLMCall) -> PrefixOverlapResult:
        """计算两个调用之间的 prefix 重叠"""
        tokens_a = call_a.prompt_tokens
        tokens_b = call_b.prompt_tokens

        # 找到最长公共前缀
        shared_len = 0
        for i in range(min(len(tokens_a), len(tokens_b))):
            if tokens_a[i] == tokens_b[i]:
                shared_len += 1
            else:
                break

        min_len = min(len(tokens_a), len(tokens_b))
        overlap_ratio = shared_len / min_len if min_len > 0 else 0

        return PrefixOverlapResult(
            call_a=f"{call_a.node_name}@hop{call_a.hop}",
            call_b=f"{call_b.node_name}@hop{call_b.hop}",
            shared_prefix_tokens=shared_len,
            total_tokens_a=len(tokens_a),
            total_tokens_b=len(tokens_b),
            overlap_ratio=overlap_ratio
        )

    def analyze_intra_request_overlap(self, request_id: str) -> Dict[str, Any]:
        """分析单个请求内的 prefix 重叠"""
        request_calls = [c for c in self.calls if c.request_id == request_id]

        if len(request_calls) < 2:
            return {"overlaps": [], "summary": {}}

        overlaps = []
        for i in range(len(request_calls)):
            for j in range(i + 1, len(request_calls)):
                overlap = self.compute_prefix_overlap(request_calls[i], request_calls[j])
                overlaps.append(asdict(overlap))

        # 统计
        avg_overlap = sum(o["overlap_ratio"] for o in overlaps) / len(overlaps) if overlaps else 0
        max_overlap = max(o["overlap_ratio"] for o in overlaps) if overlaps else 0
        total_shared_tokens = sum(o["shared_prefix_tokens"] for o in overlaps)

        return {
            "request_id": request_id,
            "num_calls": len(request_calls),
            "overlaps": overlaps,
            "summary": {
                "avg_overlap_ratio": avg_overlap,
                "max_overlap_ratio": max_overlap,
                "total_shared_tokens": total_shared_tokens,
            }
        }

    def analyze_cross_request_overlap(self) -> Dict[str, Any]:
        """分析跨请求的 prefix 重叠（相同节点类型）"""
        # 按节点类型分组
        by_node = defaultdict(list)
        for call in self.calls:
            by_node[call.node_name].append(call)

        results = {}
        for node_name, calls in by_node.items():
            if len(calls) < 2:
                continue

            # 计算该节点类型跨请求的重叠
            overlaps = []
            # 只比较前几个，避免 O(n^2) 太慢
            sample_calls = calls[:min(10, len(calls))]

            for i in range(len(sample_calls)):
                for j in range(i + 1, len(sample_calls)):
                    if sample_calls[i].request_id != sample_calls[j].request_id:
                        overlap = self.compute_prefix_overlap(sample_calls[i], sample_calls[j])
                        overlaps.append(asdict(overlap))

            avg_overlap = sum(o["overlap_ratio"] for o in overlaps) / len(overlaps) if overlaps else 0

            results[node_name] = {
                "num_calls": len(calls),
                "avg_cross_request_overlap": avg_overlap,
                "sample_overlaps": overlaps[:5]  # 只保存前5个样例
            }

        return results

    def analyze_prompt_structure(self) -> Dict[str, Any]:
        """分析各节点的 prompt 结构"""
        by_node = defaultdict(list)
        for call in self.calls:
            by_node[call.node_name].append(call)

        results = {}
        for node_name, calls in by_node.items():
            token_counts = [c.token_count for c in calls]
            results[node_name] = {
                "num_calls": len(calls),
                "avg_tokens": sum(token_counts) / len(token_counts) if token_counts else 0,
                "min_tokens": min(token_counts) if token_counts else 0,
                "max_tokens": max(token_counts) if token_counts else 0,
                "std_tokens": float(np.std(token_counts)) if HAS_MATPLOTLIB and token_counts else 0,
            }

        return results

    def get_full_analysis(self) -> Dict[str, Any]:
        """获取完整分析结果"""
        # 获取所有请求 ID
        request_ids = list(set(c.request_id for c in self.calls))

        # 请求内分析
        intra_request = [self.analyze_intra_request_overlap(rid) for rid in request_ids]

        # 跨请求分析
        cross_request = self.analyze_cross_request_overlap()

        # Prompt 结构分析
        structure = self.analyze_prompt_structure()

        # 总体统计
        total_calls = len(self.calls)
        total_tokens = sum(c.token_count for c in self.calls)

        # 计算理论可复用 tokens
        all_intra_shared = sum(r["summary"].get("total_shared_tokens", 0) for r in intra_request)

        return {
            "summary": {
                "total_requests": len(request_ids),
                "total_llm_calls": total_calls,
                "total_tokens": total_tokens,
                "avg_tokens_per_call": total_tokens / total_calls if total_calls > 0 else 0,
                "theoretical_reusable_tokens_intra_request": all_intra_shared,
            },
            "prompt_structure": structure,
            "intra_request_analysis": intra_request,
            "cross_request_analysis": cross_request,
        }


def simulate_hop2rag_prompts(
    question: str,
    analyzer: KVCacheAnalyzer,
    request_id: str,
    max_hops: int = 3,
    docs_per_hop: int = 5
) -> None:
    """模拟 Hop2Rag 的 LLM 调用并记录 prompts"""

    # 模拟累积的证据
    evidence_so_far = []

    for hop in range(max_hops):
        # 1. decompose_question 调用
        evidence_text = "\n".join([f"Hop {i}: {ev}" for i, ev in enumerate(evidence_so_far)]) if evidence_so_far else "None"
        decompose_prompt = QUESTION_DECOMPOSER_PROMPT.format(
            current_hop=hop,
            original_question=question,
            evidence=evidence_text
        )
        analyzer.record_call(request_id, "decompose", hop, decompose_prompt)

        # 模拟检索到的文档
        mock_docs = "\n\n".join([
            f"Doc {i+1}: This is a mock document about {question[:50]}... containing relevant information for hop {hop}."
            for i in range(docs_per_hop)
        ])

        # 2. extract_clues 调用
        extract_prompt = CLUE_EXTRACTOR_PROMPT.format(
            question=question,
            current_hop=hop,
            documents=mock_docs
        )
        analyzer.record_call(request_id, "extract_clues", hop, extract_prompt)

        # 模拟提取的证据
        evidence_so_far.append(f"Found entity X related to the question at hop {hop}")

        # 3. decide_next_hop 调用
        evidence_text = "\n".join([f"Hop {i}: {ev}" for i, ev in enumerate(evidence_so_far)])
        decide_prompt = HOP_DECISION_PROMPT.format(
            question=question,
            current_hop=hop,
            max_hops=max_hops,
            evidence=evidence_text,
            doc_count=docs_per_hop * (hop + 1)
        )
        analyzer.record_call(request_id, "decide", hop, decide_prompt)

    # 4. generate_final 调用
    all_docs = "\n\n".join([
        f"[Doc {i+1}] Mock document content for final generation..."
        for i in range(docs_per_hop * max_hops)
    ])
    hop_history = "\n".join([
        f"Hop {i}: Query='subquery {i}', Evidence='{evidence_so_far[i]}'"
        for i in range(len(evidence_so_far))
    ])
    generate_prompt = MULTI_HOP_RAG_PROMPT.format(
        question=question,
        context=all_docs,
        hop_history=hop_history
    )
    analyzer.record_call(request_id, "generate", max_hops, generate_prompt)


def plot_overlap_heatmap(analysis: Dict[str, Any], output_path: str):
    """绘制 prefix 重叠热力图"""
    if not HAS_MATPLOTLIB:
        return

    # 收集所有请求内的重叠数据
    all_overlaps = []
    for req_analysis in analysis["intra_request_analysis"]:
        all_overlaps.extend(req_analysis["overlaps"])

    if not all_overlaps:
        print("[WARN] No overlap data to plot")
        return

    # 构建节点名列表
    node_names = ["decompose@hop0", "extract_clues@hop0", "decide@hop0",
                  "decompose@hop1", "extract_clues@hop1", "decide@hop1",
                  "decompose@hop2", "extract_clues@hop2", "decide@hop2",
                  "generate@hop3"]

    # 构建重叠矩阵
    n = len(node_names)
    matrix = np.zeros((n, n))
    counts = np.zeros((n, n))

    for overlap in all_overlaps:
        call_a = overlap["call_a"]
        call_b = overlap["call_b"]

        if call_a in node_names and call_b in node_names:
            i = node_names.index(call_a)
            j = node_names.index(call_b)
            matrix[i, j] += overlap["overlap_ratio"]
            matrix[j, i] += overlap["overlap_ratio"]
            counts[i, j] += 1
            counts[j, i] += 1

    # 平均
    with np.errstate(divide='ignore', invalid='ignore'):
        matrix = np.divide(matrix, counts, where=counts != 0)
        matrix[counts == 0] = 0

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(matrix, cmap='YlOrRd', vmin=0, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([n.replace("@", "\n") for n in node_names], rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels([n.replace("@", "\n") for n in node_names], fontsize=9)

    # 添加数值标注
    for i in range(n):
        for j in range(n):
            if matrix[i, j] > 0:
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                              ha='center', va='center', fontsize=8,
                              color='white' if matrix[i, j] > 0.5 else 'black')

    ax.set_title('Prefix Overlap Ratio Between LLM Calls\n(Higher = More KV Cache Reuse Potential)', fontsize=14)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Overlap Ratio', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved overlap heatmap: {output_path}")


def plot_token_breakdown(analysis: Dict[str, Any], output_path: str):
    """绘制各节点 token 数量分布"""
    if not HAS_MATPLOTLIB:
        return

    structure = analysis["prompt_structure"]

    nodes = list(structure.keys())
    avg_tokens = [structure[n]["avg_tokens"] for n in nodes]
    min_tokens = [structure[n]["min_tokens"] for n in nodes]
    max_tokens = [structure[n]["max_tokens"] for n in nodes]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(nodes))
    width = 0.6

    # 绘制柱状图
    bars = ax.bar(x, avg_tokens, width, color='steelblue', alpha=0.8, label='Avg Tokens')

    # 添加误差线
    yerr_lower = [avg - min_t for avg, min_t in zip(avg_tokens, min_tokens)]
    yerr_upper = [max_t - avg for avg, max_t in zip(avg_tokens, max_tokens)]
    ax.errorbar(x, avg_tokens, yerr=[yerr_lower, yerr_upper], fmt='none', color='black', capsize=5)

    # 添加数值标注
    for bar, avg in zip(bars, avg_tokens):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{avg:.0f}', ha='center', va='bottom', fontsize=11)

    ax.set_xlabel('Node Type', fontsize=12)
    ax.set_ylabel('Token Count', fontsize=12)
    ax.set_title('Prompt Token Distribution by Node Type\n(Error bars show min-max range)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(nodes, rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved token breakdown: {output_path}")


def plot_cache_reuse_potential(analysis: Dict[str, Any], output_path: str):
    """绘制 KV Cache 复用潜力分析图"""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：请求内复用潜力
    ax1 = axes[0]
    intra_data = analysis["intra_request_analysis"]

    if intra_data:
        avg_overlaps = [r["summary"]["avg_overlap_ratio"] for r in intra_data if r["summary"]]
        max_overlaps = [r["summary"]["max_overlap_ratio"] for r in intra_data if r["summary"]]

        x = range(len(avg_overlaps))
        ax1.bar([i - 0.2 for i in x], avg_overlaps, 0.4, label='Avg Overlap', color='steelblue', alpha=0.8)
        ax1.bar([i + 0.2 for i in x], max_overlaps, 0.4, label='Max Overlap', color='coral', alpha=0.8)

        ax1.set_xlabel('Request Index', fontsize=12)
        ax1.set_ylabel('Overlap Ratio', fontsize=12)
        ax1.set_title('Intra-Request KV Cache Reuse Potential', fontsize=13)
        ax1.legend()
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3, axis='y')

    # 右图：跨请求复用潜力（按节点类型）
    ax2 = axes[1]
    cross_data = analysis["cross_request_analysis"]

    if cross_data:
        nodes = list(cross_data.keys())
        overlaps = [cross_data[n]["avg_cross_request_overlap"] for n in nodes]

        bars = ax2.bar(nodes, overlaps, color='forestgreen', alpha=0.8)

        for bar, val in zip(bars, overlaps):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=11)

        ax2.set_xlabel('Node Type', fontsize=12)
        ax2.set_ylabel('Avg Overlap Ratio', fontsize=12)
        ax2.set_title('Cross-Request KV Cache Reuse Potential\n(Same node type, different requests)', fontsize=13)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved cache reuse potential: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="KV Cache Reuse Potential Analysis for Hop2Rag"
    )
    parser.add_argument("--limit", type=int, default=10, help="Number of questions to analyze")
    parser.add_argument("--max-hops", type=int, default=3, help="Max hops per question")
    parser.add_argument("--docs-per-hop", type=int, default=5, help="Simulated docs per hop")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Tokenizer model")
    parser.add_argument("--skip-plots", action="store_true", help="Skip visualization")

    args = parser.parse_args()

    print("=" * 70)
    print("KV Cache Reuse Potential Analysis")
    print("=" * 70)
    print(f"Questions:     {args.limit}")
    print(f"Max Hops:      {args.max_hops}")
    print(f"Docs per Hop:  {args.docs_per_hop}")
    print(f"Tokenizer:     {args.model}")
    print("=" * 70)
    print()

    # 创建输出目录
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 初始化分析器
    analyzer = KVCacheAnalyzer(model_name=args.model)

    # 模拟问题
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

    # 扩展问题列表
    questions = (sample_questions * ((args.limit // len(sample_questions)) + 1))[:args.limit]

    # 模拟 LLM 调用
    print("Simulating LLM calls...")
    for i, question in enumerate(questions):
        request_id = f"req_{i:04d}"
        simulate_hop2rag_prompts(
            question=question,
            analyzer=analyzer,
            request_id=request_id,
            max_hops=args.max_hops,
            docs_per_hop=args.docs_per_hop
        )
        print(f"  [{i+1}/{len(questions)}] Processed: {question[:50]}...")

    print()
    print("Analyzing prefix overlaps...")
    analysis = analyzer.get_full_analysis()

    # 保存结果
    output_json = RESULTS_DIR / "prefix_overlap_analysis.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        # 转换不可序列化的字段
        serializable = {
            "summary": analysis["summary"],
            "prompt_structure": analysis["prompt_structure"],
            "cross_request_analysis": analysis["cross_request_analysis"],
            "intra_request_analysis": [
                {
                    "request_id": r["request_id"],
                    "num_calls": r["num_calls"],
                    "summary": r["summary"],
                    "overlaps": r["overlaps"][:10]  # 只保存前10个
                }
                for r in analysis["intra_request_analysis"]
            ]
        }
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved analysis: {output_json}")

    # 打印摘要
    print()
    print("=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    summary = analysis["summary"]
    print(f"Total Requests:        {summary['total_requests']}")
    print(f"Total LLM Calls:       {summary['total_llm_calls']}")
    print(f"Total Tokens:          {summary['total_tokens']}")
    print(f"Avg Tokens per Call:   {summary['avg_tokens_per_call']:.1f}")
    print()

    print("Prompt Structure by Node:")
    for node, stats in analysis["prompt_structure"].items():
        print(f"  {node:20s}: avg={stats['avg_tokens']:.0f}, "
              f"min={stats['min_tokens']}, max={stats['max_tokens']}")
    print()

    print("Cross-Request Overlap (same node type):")
    for node, data in analysis["cross_request_analysis"].items():
        print(f"  {node:20s}: {data['avg_cross_request_overlap']:.2%}")
    print()

    # 绘图
    if not args.skip_plots and HAS_MATPLOTLIB:
        print("Generating visualizations...")

        plot_overlap_heatmap(analysis, str(RESULTS_DIR / "prefix_overlap_heatmap.png"))
        plot_token_breakdown(analysis, str(RESULTS_DIR / "token_breakdown.png"))
        plot_cache_reuse_potential(analysis, str(RESULTS_DIR / "cache_reuse_potential.png"))

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
