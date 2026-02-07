"""
CRag2 Data Collector Test
测试 LanggraphMonitor 收集 LLM 调用数据（prompt 和 response）
"""

import argparse
import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# =============================
# 路径设置
# =============================

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# =============================
# 导入 CRag2
# =============================

from Rag.c_rag2 import (
    run_c_rag2,
    get_llm_calls,
    get_prompt_token_distribution,
    get_performance_summary,
    clear_performance_records,
)

# =============================
# 环境变量
# =============================

FAISS_DIR = os.environ.get("AGRAG_FAISS_DIR", "")
RESULTS_DIR = ROOT / "tests" / "results" / "crag2_datacollector"


# =============================
# 数据收集和保存
# =============================

def run_single_query(question: str, retrieval_k: int = 15) -> Dict[str, Any]:
    """运行单个查询并收集数据"""
    clear_performance_records()

    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")

    # 运行查询
    result = run_c_rag2(question, retrieval_k=retrieval_k)

    # 获取 LLM 调用记录
    llm_calls = get_llm_calls()

    # 获取统计信息
    token_dist = get_prompt_token_distribution()
    summary = get_performance_summary()

    print(f"\n✓ Answer: {result['answer'][:200]}...")
    print(f"✓ LLM Calls: {len(llm_calls)}")
    print(f"✓ Total Prompt Tokens: {token_dist.get('total_tokens', 0)}")

    return {
        "question": question,
        "answer": result["answer"],
        "metadata": result["metadata"],
        "llm_calls": llm_calls,
        "token_distribution": token_dist,
        "performance_summary": summary,
    }


def save_results(results: List[Dict[str, Any]], output_dir: Path):
    """保存收集的数据"""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存完整数据
    full_data_path = output_dir / f"llm_calls_{timestamp}.json"
    with open(full_data_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Full data saved to: {full_data_path}")

    # 保存 LLM calls 摘要（只包含 prompt 和 response）
    llm_summary = []
    for result in results:
        for call in result["llm_calls"]:
            llm_summary.append({
                "question": result["question"],
                "node_name": call.get("node_name", "unknown"),
                "prompt": call.get("prompt", ""),
                "response": call.get("response", ""),
                "prompt_tokens": call.get("prompt_tokens", 0),
                "response_tokens": call.get("response_tokens", 0),
            })

    summary_path = output_dir / f"llm_summary_{timestamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(llm_summary, f, ensure_ascii=False, indent=2)

    print(f"✓ LLM summary saved to: {summary_path}")

    # 生成统计报告
    report_path = output_dir / f"report_{timestamp}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("CRag2 Data Collection Report\n")
        f.write("="*60 + "\n\n")

        f.write(f"Total Questions: {len(results)}\n")
        f.write(f"Total LLM Calls: {sum(len(r['llm_calls']) for r in results)}\n\n")

        for i, result in enumerate(results, 1):
            f.write(f"\n{'='*60}\n")
            f.write(f"Question {i}: {result['question']}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Answer: {result['answer'][:200]}...\n\n")

            f.write(f"LLM Calls: {len(result['llm_calls'])}\n")
            token_dist = result['token_distribution']
            f.write(f"Total Prompt Tokens: {token_dist.get('total_tokens', 0)}\n")
            f.write(f"Avg Prompt Tokens: {token_dist.get('mean_tokens', 0):.1f}\n\n")

            f.write("LLM Call Details:\n")
            for j, call in enumerate(result['llm_calls'], 1):
                f.write(f"\n  Call {j} - Node: {call.get('node_name', 'unknown')}\n")
                f.write(f"  Prompt Tokens: {call.get('prompt_tokens', 0)}\n")
                f.write(f"  Response Tokens: {call.get('response_tokens', 0)}\n")
                f.write(f"  Prompt Preview: {call.get('prompt', '')[:100]}...\n")
                f.write(f"  Response Preview: {call.get('response', '')[:100]}...\n")

    print(f"✓ Report saved to: {report_path}")


# =============================
# 主函数
# =============================

def main():
    ap = argparse.ArgumentParser(description="CRag2 Data Collector Test")
    ap.add_argument("--questions", type=str, nargs="+", help="Questions to test")
    ap.add_argument("--limit", type=int, default=3, help="Number of default questions to use")
    ap.add_argument("--retrieval-k", type=int, default=15, help="Number of documents to retrieve")
    args = ap.parse_args()

    if not FAISS_DIR:
        print("[ERROR] Set AGRAG_FAISS_DIR environment variable")
        sys.exit(1)

    # 准备问题
    if args.questions:
        questions = args.questions
    else:
        default_questions = [
            "What is machine learning?",
            "Who invented the telephone?",
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "What are the main causes of climate change?",
        ]
        questions = default_questions[:args.limit]

    print("="*60)
    print("CRag2 Data Collector Test")
    print("="*60)
    print(f"Questions: {len(questions)}")
    print(f"Retrieval K: {args.retrieval_k}")
    print()

    # 运行查询并收集数据
    results = []
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] Processing...")
        result = run_single_query(question, retrieval_k=args.retrieval_k)
        results.append(result)

    # 保存结果
    print(f"\n{'='*60}")
    print("Saving results...")
    print(f"{'='*60}")
    save_results(results, RESULTS_DIR)

    # 打印总结
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    total_llm_calls = sum(len(r['llm_calls']) for r in results)
    total_prompt_tokens = sum(r['token_distribution'].get('total_tokens', 0) for r in results)

    print(f"Total Questions: {len(results)}")
    print(f"Total LLM Calls: {total_llm_calls}")
    print(f"Total Prompt Tokens: {total_prompt_tokens}")
    print(f"Avg LLM Calls per Question: {total_llm_calls / len(results):.1f}")
    print(f"Avg Prompt Tokens per Question: {total_prompt_tokens / len(results):.1f}")

    print(f"\n✓ All results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
