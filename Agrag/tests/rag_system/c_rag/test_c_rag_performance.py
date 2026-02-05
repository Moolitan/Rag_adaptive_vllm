"""
CRag Performance Test with DataCollector (FAISS backend)

- 使用 Corrective RAG（CRag）
- 向量库：FAISS（通过 AGRAG_FAISS_DIR）
- 性能采集：runner.performancemonitor.DataCollector
- LLM 监控：VLLMMonitor
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

# =============================
# 路径设置
# =============================

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# =============================
# 导入 CRag & 监控
# =============================

from runner.VLLMMonitor import VLLMMonitor

from Rag.c_rag_performancy import (
    run_c_rag,
    get_llm_calls,
    get_prompt_token_distribution,
    get_performance_records,
    get_performance_summary,
    clear_performance_records,
)

# =============================
# 环境变量
# =============================

FAISS_DIR = os.environ.get("AGRAG_FAISS_DIR", "")

RESULTS_DIR = ROOT / "tests" / "results" / "crag_performance"


# =============================
# 数据结构
# =============================

@dataclass
class PerformanceResult:
    """单次请求的性能结果"""
    question: str
    answer: str

    total_latency_sec: float

    llm_calls: int
    total_llm_latency_sec: float
    total_input_tokens: int
    total_output_tokens: int

    node_executions: int
    used_web_search: bool
    web_search_rounds: int

    records: List[Dict[str, Any]]
    llm_call_details: List[Dict[str, Any]]  # 每个 LLM call 的 prompt 和 response


# =============================
# 单条测试
# =============================

def run_single_test(question: str) -> PerformanceResult:
    """运行单个 CRag 查询并收集性能数据"""
    clear_performance_records()

    start = time.time()
    result = run_c_rag(question)
    total_latency = time.time() - start

    # 获取节点级别的执行记录
    records = get_performance_records()
    # 获取 LLM call 的详细记录（包含 prompt 和 response）
    llm_call_details = get_llm_calls()
    # 获取性能摘要
    summary = get_performance_summary()

    # 从节点记录中筛选
    node_records = [r for r in records if r.get("event") == "node_execution"]

    # 从 llm_call_details 计算 token 统计
    total_input_tokens = sum(call.get("prompt_tokens", 0) for call in llm_call_details)
    total_output_tokens = sum(call.get("response_tokens", 0) for call in llm_call_details)

    # 从 summary 获取 LLM 延迟
    total_llm_latency = summary.get("llm_latency_sec", 0)

    metadata = result.get("metadata", {})

    return PerformanceResult(
        question=question,
        answer=result.get("answer", ""),

        total_latency_sec=total_latency,

        llm_calls=len(llm_call_details),
        total_llm_latency_sec=total_llm_latency,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,

        node_executions=len(node_records),
        used_web_search=bool(metadata.get("used_web_search", False)),
        web_search_rounds=int(metadata.get("web_search_rounds", 0)),

        records=list(records),
        llm_call_details=llm_call_details
    )


# =============================
# 批量测试
# =============================

def run_benchmark(
    questions: List[str],
    verbose: bool = True
) -> List[PerformanceResult]:

    results = []

    for i, q in enumerate(questions):
        if verbose:
            print(f"[{i+1}/{len(questions)}] {q[:80]}")

        perf = run_single_test(q)
        results.append(perf)

        if verbose:
            print(
                f"  Latency: {perf.total_latency_sec:.2f}s | "
                f"LLM calls: {perf.llm_calls} | "
                f"Tokens: {perf.total_input_tokens}+{perf.total_output_tokens} | "
                f"WebSearch: {perf.used_web_search} ({perf.web_search_rounds})"
            )

    return results


# =============================
# 统计计算
# =============================

def compute_stats(results: List[PerformanceResult]) -> Dict[str, Any]:
    if not results:
        return {}

    latencies = [r.total_latency_sec for r in results]
    llm_latencies = [r.total_llm_latency_sec for r in results]
    llm_calls = [r.llm_calls for r in results]
    input_tokens = [r.total_input_tokens for r in results]
    output_tokens = [r.total_output_tokens for r in results]
    web_search_used = [r.used_web_search for r in results]
    web_rounds = [r.web_search_rounds for r in results]

    return {
        "total_requests": len(results),

        "latency_mean": float(np.mean(latencies)),
        "latency_p50": float(np.percentile(latencies, 50)),
        "latency_p95": float(np.percentile(latencies, 95)),

        "llm_latency_mean": float(np.mean(llm_latencies)),
        "llm_calls_mean": float(np.mean(llm_calls)),

        "input_tokens_mean": float(np.mean(input_tokens)),
        "output_tokens_mean": float(np.mean(output_tokens)),

        "web_search_rate": float(np.mean(web_search_used)),
        "web_search_rounds_mean": float(np.mean(web_rounds)),
    }


# =============================
# 保存结果
# =============================

def save_results(
    results: List[PerformanceResult],
    stats: Dict[str, Any],
    output_dir: Path
):
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "performance_results.json", "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)

    with open(output_dir / "performance_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_dir}")


# =============================
# 主函数
# =============================

def main():
    ap = argparse.ArgumentParser(description="CRag Performance Test (FAISS + DataCollector)")
    ap.add_argument("--limit", type=int, default=5, help="Number of questions")
    ap.add_argument("--monitor-interval", type=float, default=0.5)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not FAISS_DIR:
        print("[ERROR] Set AGRAG_FAISS_DIR environment variable")
        sys.exit(1)

    questions = [
        # "What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?",
        # "The director of the romantic comedy Big Stone Gap is based in what New York city?",
        # "2014 S/S is the debut album of a South Korean boy group that was formed by who?",
        # "The arena where the Lewiston Maineiacs played their home games can seat how many people?",
        # "What screenwriter with credits for Evolution co-wrote a film starring Nicolas Cage and Téa Leoni?",
        # "Which city is the capital of China?"
    ]

    questions = questions[:min(args.limit, len(questions))]

    print("=" * 60)
    print("CRag Performance Test (FAISS)")
    print("=" * 60)
    print(f"Questions: {len(questions)}")
    print()

    monitor = VLLMMonitor(
        url="http://localhost:8000/metrics",
        interval=args.monitor_interval,
        csv_path=RESULTS_DIR / "vllm_metrics.csv",
        flush_every=1
    )

    monitor.start()
    try:
        results = run_benchmark(questions, args.verbose)
        stats = compute_stats(results)
    finally:
        monitor.stop()

    print("\n" + "=" * 60)
    print("Statistics")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

    # path = [
    #     r["node"]
    #     for r in records
    #     if r["event"] == "node_execution"
    # ]

    # print(" → ".join(path))

    save_results(results, stats, RESULTS_DIR)


if __name__ == "__main__":
    main()
