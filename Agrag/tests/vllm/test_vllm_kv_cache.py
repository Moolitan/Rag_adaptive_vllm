"""
vLLM KV Cache Usage Test

测试 vLLM 的 KV Cache 使用情况：
- 使用固定的输入文本
- 反复推理 10000 次
- 监控 GPU KV Cache usage 的变化
"""

import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
import json

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from openai import OpenAI
from Agrag.runner.VLLMMonitor import VLLMMonitor


# vLLM 配置
VLLM_API_BASE = "http://localhost:8000/v1"
VLLM_API_KEY = "EMPTY"
VLLM_MODEL_NAME = "Qwen2.5"


def create_client() -> OpenAI:
    """创建 OpenAI 客户端（连接到 vLLM）"""
    return OpenAI(
        api_key=VLLM_API_KEY,
        base_url=VLLM_API_BASE,
    )


def run_inference(client: OpenAI, prompt: str, max_tokens: int = 50) -> Dict[str, Any]:
    """
    运行单次推理

    Args:
        client: OpenAI 客户端
        prompt: 输入提示词
        max_tokens: 最大生成 token 数

    Returns:
        包含响应信息的字典
    """
    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model=VLLM_MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.0,  # 确保结果一致性
        )

        latency = time.time() - start_time

        return {
            "success": True,
            "latency": latency,
            "response": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "latency": time.time() - start_time,
        }


def run_benchmark(
    num_iterations: int = 10000,
    prompt: str = None,
    max_tokens: int = 50,
    monitor_interval: float = 0.5,
    output_dir: Path = None,
    batch_size: int = 1,
    delay_between_batches: float = 0.0,
) -> Dict[str, Any]:
    """
    运行 vLLM KV Cache 基准测试

    Args:
        num_iterations: 推理次数
        prompt: 输入提示词
        max_tokens: 最大生成 token 数
        monitor_interval: 监控采样间隔（秒）
        output_dir: 输出目录
        batch_size: 每批次请求数
        delay_between_batches: 批次间延迟（秒）

    Returns:
        测试结果统计
    """
    if prompt is None:
        prompt = (
            "Please provide a comprehensive explanation of machine learning and its applications in modern technology. "
            "Machine learning is a subset of artificial intelligence that enables computers to learn from data "
            "without being explicitly programmed. It has revolutionized various industries including healthcare, "
            "finance, transportation, and entertainment. The fundamental principle behind machine learning is to "
            "develop algorithms that can identify patterns in data and make predictions or decisions based on those patterns. "
            "There are three main types of machine learning: supervised learning, where the algorithm learns from labeled data; "
            "unsupervised learning, where the algorithm finds hidden patterns in unlabeled data; and reinforcement learning, "
            "where the algorithm learns through trial and error by receiving rewards or penalties. Deep learning, a subset "
            "of machine learning, uses artificial neural networks with multiple layers to process complex data such as images, "
            "speech, and natural language. Popular machine learning frameworks include TensorFlow, PyTorch, and scikit-learn. "
            "Applications of machine learning range from recommendation systems used by Netflix and Amazon, to autonomous vehicles "
            "developed by Tesla and Waymo, to medical diagnosis systems that can detect diseases from medical images with high accuracy. "
            "As machine learning continues to advance, it raises important questions about ethics, privacy, bias, and the future "
            "of work. Understanding these technologies is crucial for anyone working in the modern tech industry."
        )

    if output_dir is None:
        output_dir = ROOT / "tests" / "results" / "vllm_kv_cache"

    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建客户端
    client = create_client()

    # 启动监控
    csv_path = output_dir / "vllm_metrics.csv"
    monitor = VLLMMonitor(
        url="http://localhost:8000/metrics",
        interval=monitor_interval,
        csv_path=str(csv_path),
        flush_every=1,
    )

    print("=" * 80)
    print("vLLM KV Cache Usage Test")
    print("=" * 80)
    print(f"Iterations: {num_iterations}")
    print(f"Batch size: {batch_size}")
    print(f"Delay between batches: {delay_between_batches}s")
    print(f"Max tokens per request: {max_tokens}")
    print(f"Monitor interval: {monitor_interval}s")
    print(f"Output directory: {output_dir}")
    print(f"Prompt length: {len(prompt)} characters")
    print(f"Prompt preview: {prompt[:100]}...")
    print("=" * 80)
    print()

    # 启动监控
    monitor.start()

    # 运行测试
    results = []
    start_time = time.time()

    try:
        num_batches = (num_iterations + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, num_iterations)
            current_batch_size = batch_end - batch_start

            # 执行当前批次的请求
            for i in range(current_batch_size):
                iteration = batch_start + i

                result = run_inference(client, prompt, max_tokens)
                results.append(result)

                if (iteration + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    avg_latency = sum(r.get("latency", 0) for r in results) / len(results)
                    success_rate = sum(1 for r in results if r.get("success", False)) / len(results) * 100

                    print(f"[{iteration + 1}/{num_iterations}] "
                          f"Elapsed: {elapsed:.1f}s, "
                          f"Avg latency: {avg_latency:.3f}s, "
                          f"Success rate: {success_rate:.1f}%")

            # 批次间延迟
            if delay_between_batches > 0 and batch_idx < num_batches - 1:
                time.sleep(delay_between_batches)

        total_time = time.time() - start_time

    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        total_time = time.time() - start_time
    finally:
        # 停止监控
        monitor.stop()

    # 计算统计
    successful_results = [r for r in results if r.get("success", False)]
    failed_results = [r for r in results if not r.get("success", False)]

    stats = {
        "total_iterations": len(results),
        "successful": len(successful_results),
        "failed": len(failed_results),
        "success_rate": len(successful_results) / len(results) * 100 if results else 0,
        "total_time_sec": total_time,
        "throughput_req_per_sec": len(results) / total_time if total_time > 0 else 0,
    }

    if successful_results:
        latencies = [r["latency"] for r in successful_results]
        stats.update({
            "avg_latency_sec": sum(latencies) / len(latencies),
            "min_latency_sec": min(latencies),
            "max_latency_sec": max(latencies),
            "p50_latency_sec": sorted(latencies)[len(latencies) // 2],
            "p95_latency_sec": sorted(latencies)[int(len(latencies) * 0.95)],
            "p99_latency_sec": sorted(latencies)[int(len(latencies) * 0.99)],
        })

        # Token 统计
        if "usage" in successful_results[0]:
            prompt_tokens = [r["usage"]["prompt_tokens"] for r in successful_results]
            completion_tokens = [r["usage"]["completion_tokens"] for r in successful_results]
            total_tokens = [r["usage"]["total_tokens"] for r in successful_results]

            stats.update({
                "avg_prompt_tokens": sum(prompt_tokens) / len(prompt_tokens),
                "avg_completion_tokens": sum(completion_tokens) / len(completion_tokens),
                "avg_total_tokens": sum(total_tokens) / len(total_tokens),
                "total_prompt_tokens": sum(prompt_tokens),
                "total_completion_tokens": sum(completion_tokens),
                "total_tokens_processed": sum(total_tokens),
            })

    # 保存结果
    print("\n" + "=" * 80)
    print("Test Results")
    print("=" * 80)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # 保存统计数据
    stats_file = output_dir / "test_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    # 保存详细结果（可选，数据量大时可能很大）
    if len(results) <= 1000:  # 只保存小规模测试的详细结果
        results_file = output_dir / "detailed_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {results_file}")

    print(f"Statistics saved to: {stats_file}")
    print(f"vLLM metrics saved to: {csv_path}")
    print("=" * 80)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Test vLLM KV Cache usage with repeated inference"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10000,
        help="Number of inference iterations (default: 10000)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt text (default: built-in prompt)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate per request (default: 50)"
    )
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=0.5,
        help="Monitor sampling interval in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: tests/results/vllm_kv_cache)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of requests per batch (default: 1)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay between batches in seconds (default: 0.0)"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None

    try:
        run_benchmark(
            num_iterations=args.iterations,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            monitor_interval=args.monitor_interval,
            output_dir=output_dir,
            batch_size=args.batch_size,
            delay_between_batches=args.delay,
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
