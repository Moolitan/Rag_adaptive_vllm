#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM-only baseline benchmark.

Purpose: Establish performance and memory boundaries of the inference backend.
This baseline is used to explain RAG results, not replaced by them.

Focus:
- Pure inference performance (no retrieval, no RAG logic)
- TTFT, ITL, E2E latency (P50/P95/P99)
- Context length scaling
- Concurrency behavior
- GPU memory usage (if available)

Usage:
    # Single context length test
    python tests/vllm_baseline.py --context-length 2000 --num-requests 50

    # Sweep context lengths
    python tests/vllm_baseline.py --context-sweep 500,1000,2000,4000 --num-requests 50

    # Test concurrency
    python tests/vllm_baseline.py --context-length 2000 --concurrency 4 --num-requests 50
"""

from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import concurrent.futures
import threading

ROOT = Path(__file__).resolve().parents[2]  # Agrag directory
sys.path.insert(0, str(ROOT))

from core.config import get_vllm_chat, VLLM_API_BASE

# Import real question generator
from real_questions import get_question_for_context_and_request


@dataclass
class InferenceTrace:
    """Single inference request trace"""
    request_id: str

    # Request configuration
    context_length: int        # Input context length (tokens)
    max_tokens: int           # Max generation tokens

    # Timing (seconds)
    submit_time: float
    start_time: float
    end_time: float

    # Latency breakdown
    ttft: float = 0.0         # Time to first token (if available)
    itl: float = 0.0          # Inter-token latency (if available)
    e2e_latency: float = 0.0  # End-to-end latency

    # Tokens
    prompt_tokens: int = 0     # Actual prompt tokens used
    completion_tokens: int = 0 # Generated tokens
    total_tokens: int = 0      # Total tokens

    # System state
    concurrent_requests: int = 0

    # Result
    success: bool = True
    error: str = ""


class VLLMBaseline:
    """vLLM-only baseline benchmark"""

    def __init__(self):
        self.llm = get_vllm_chat(json_mode=False, temperature=0)
        self.api_base = VLLM_API_BASE

        # Shared state for concurrency tracking
        self.active_requests = 0
        self.lock = threading.Lock()

    def generate_prompt(self, context_length: int, request_id: int) -> str:
        """
        Generate a real, diverse question for the given context length and request ID.

        Uses real_questions module to ensure:
        - Each request gets a unique question (no KV cache pollution)
        - Questions are realistic and diverse
        - Same request_id gets same question (reproducibility)
        - Different context lengths get completely different question pools
        """
        return get_question_for_context_and_request(context_length, request_id)

    def run_single_inference(
        self,
        req_id: str,
        context_length: int,
        max_tokens: int,
        submit_time: float
    ) -> InferenceTrace:
        """Run a single inference request and collect trace"""

        with self.lock:
            self.active_requests += 1
            concurrent = self.active_requests

        start_time = time.time()

        trace = InferenceTrace(
            request_id=req_id,
            context_length=context_length,
            max_tokens=max_tokens,
            submit_time=submit_time,
            start_time=start_time,
            end_time=0.0,
            e2e_latency=0.0,
            concurrent_requests=concurrent
        )

        try:
            # Generate real, unique prompt
            # Extract request number from req_id (e.g., "req_0" -> 0)
            request_num = int(req_id.split('_')[1]) if '_' in req_id else 0
            prompt = self.generate_prompt(context_length, request_num)

            # Call vLLM
            messages = [{"role": "user", "content": prompt}]

            # Time the inference
            inference_start = time.time()
            response = self.llm.invoke(messages, max_tokens=max_tokens)
            inference_end = time.time()

            # Extract response
            if hasattr(response, 'content'):
                output = response.content
            else:
                output = str(response)

            # Try to get token counts from response metadata
            if hasattr(response, 'response_metadata'):
                usage = response.response_metadata.get('token_usage', {})
                trace.prompt_tokens = usage.get('prompt_tokens', 0)
                trace.completion_tokens = usage.get('completion_tokens', 0)
                trace.total_tokens = usage.get('total_tokens', 0)

            # If token counts not available, estimate
            if trace.prompt_tokens == 0:
                trace.prompt_tokens = len(prompt.split()) // 1  # rough estimate
                trace.completion_tokens = len(output.split()) // 1
                trace.total_tokens = trace.prompt_tokens + trace.completion_tokens

            trace.end_time = inference_end
            trace.e2e_latency = inference_end - inference_start
            trace.success = True

            # Note: TTFT and ITL require streaming API, not available in simple invoke
            # For more detailed metrics, would need to use vLLM's /v1/completions with stream=True

        except Exception as e:
            trace.end_time = time.time()
            trace.e2e_latency = trace.end_time - start_time
            trace.success = False
            trace.error = str(e)

        finally:
            with self.lock:
                self.active_requests -= 1

        return trace

    def run_serial(
        self,
        num_requests: int,
        context_length: int,
        max_tokens: int,
        verbose: bool = True
    ) -> List[InferenceTrace]:
        """Run requests serially (concurrency=1)"""
        traces = []

        for i in range(num_requests):
            if verbose:
                print(f"[{i+1}/{num_requests}] Request {i} (context={context_length} tokens)...")

            submit_time = time.time()
            trace = self.run_single_inference(
                req_id=f"req_{i}",
                context_length=context_length,
                max_tokens=max_tokens,
                submit_time=submit_time
            )
            traces.append(trace)

            if verbose:
                print(f"  ✓ Latency: {trace.e2e_latency:.3f}s, Tokens: {trace.prompt_tokens}/{trace.completion_tokens}")

        return traces

    def run_concurrent(
        self,
        num_requests: int,
        context_length: int,
        max_tokens: int,
        max_workers: int,
        verbose: bool = True
    ) -> List[InferenceTrace]:
        """Run requests with controlled concurrency"""
        traces = []
        submit_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i in range(num_requests):
                future = executor.submit(
                    self.run_single_inference,
                    f"req_{i}",
                    context_length,
                    max_tokens,
                    submit_time
                )
                futures.append(future)

            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                if verbose:
                    print(f"[{i+1}/{num_requests}] Request completed")
                trace = future.result()
                traces.append(trace)

        return traces


def compute_statistics(traces: List[InferenceTrace]) -> Dict[str, Any]:
    """Compute statistics from traces"""

    if not traces:
        return {}

    latencies = [t.e2e_latency for t in traces if t.success]
    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)

    if n == 0:
        return {"error": "No successful requests"}

    # Percentiles
    p50 = latencies_sorted[int(n * 0.50)] if n > 0 else 0
    p95 = latencies_sorted[int(n * 0.95)] if n > 0 else 0
    p99 = latencies_sorted[int(n * 0.99)] if n > 0 else 0

    # Timing
    total_time = max(t.end_time for t in traces) - min(t.submit_time for t in traces)
    throughput = len(traces) / total_time if total_time > 0 else 0

    # Tokens
    avg_prompt_tokens = sum(t.prompt_tokens for t in traces) / len(traces)
    avg_completion_tokens = sum(t.completion_tokens for t in traces) / len(traces)

    return {
        "total_requests": len(traces),
        "successful_requests": sum(1 for t in traces if t.success),
        "failed_requests": sum(1 for t in traces if not t.success),

        # Latency (seconds)
        "latency_mean": sum(latencies) / n,
        "latency_median": p50,
        "latency_p95": p95,
        "latency_p99": p99,
        "latency_min": min(latencies),
        "latency_max": max(latencies),

        # Throughput
        "total_time_sec": total_time,
        "throughput_req_per_sec": throughput,

        # Tokens
        "avg_prompt_tokens": avg_prompt_tokens,
        "avg_completion_tokens": avg_completion_tokens,

        # Context
        "context_length": traces[0].context_length if traces else 0,
        "max_tokens": traces[0].max_tokens if traces else 0,
    }


def save_traces(traces: List[InferenceTrace], output_path: str):
    """Save traces to JSON file"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([asdict(t) for t in traces], f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser(description="vLLM-only baseline benchmark")

    # Basic configuration
    ap.add_argument(
        "--context-length",
        type=int,
        default=2000,
        help="Context length in tokens (default: 2000)"
    )
    ap.add_argument(
        "--context-sweep",
        type=str,
        help="Sweep context lengths (comma-separated, e.g., '500,1000,2000,4000')"
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Max generation tokens (default: 50)"
    )
    ap.add_argument(
        "--num-requests",
        type=int,
        default=50,
        help="Number of requests per test (default: 50)"
    )
    ap.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent requests (1=serial, default: 1)"
    )

    # Output
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "tests" / "results" / "vllm_baseline"),
        help="Output directory for results"
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = ap.parse_args()

    print("=" * 80)
    print("vLLM-only Baseline Benchmark")
    print("=" * 80)
    print(f"API Base: {VLLM_API_BASE}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Requests: {args.num_requests}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Output: {args.out_dir}")
    print("=" * 80)
    print()

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine context lengths to test
    if args.context_sweep:
        context_lengths = [int(x.strip()) for x in args.context_sweep.split(',')]
        print(f"Context sweep: {context_lengths}")
    else:
        context_lengths = [args.context_length]
        print(f"Context length: {args.context_length}")

    print()

    # Run benchmarks
    bench = VLLMBaseline()
    all_results = []

    for ctx_len in context_lengths:
        print(f"\n{'='*80}")
        print(f"Testing context length: {ctx_len} tokens")
        print(f"{'='*80}\n")

        start_time = time.time()

        if args.concurrency == 1:
            traces = bench.run_serial(
                num_requests=args.num_requests,
                context_length=ctx_len,
                max_tokens=args.max_tokens,
                verbose=args.verbose
            )
        else:
            traces = bench.run_concurrent(
                num_requests=args.num_requests,
                context_length=ctx_len,
                max_tokens=args.max_tokens,
                max_workers=args.concurrency,
                verbose=args.verbose
            )

        elapsed = time.time() - start_time

        print(f"\nCompleted in {elapsed:.2f}s\n")

        # Compute statistics
        stats = compute_statistics(traces)
        stats["concurrency"] = args.concurrency
        all_results.append(stats)

        # Print results
        print("=" * 80)
        print(f"RESULTS: Context={ctx_len} tokens, Concurrency={args.concurrency}")
        print("=" * 80)
        print(f"Total requests:     {stats['total_requests']}")
        print(f"Successful:         {stats['successful_requests']}")
        print(f"Failed:             {stats['failed_requests']}")
        print()
        print(f"Latency (mean):     {stats['latency_mean']:.3f}s")
        print(f"Latency (median):   {stats['latency_median']:.3f}s")
        print(f"Latency (P95):      {stats['latency_p95']:.3f}s")
        print(f"Latency (P99):      {stats['latency_p99']:.3f}s")
        print(f"Latency (min/max):  {stats['latency_min']:.3f}s / {stats['latency_max']:.3f}s")
        print()
        print(f"Throughput:         {stats['throughput_req_per_sec']:.2f} req/s")
        print(f"Total time:         {stats['total_time_sec']:.2f}s")
        print()
        print(f"Avg prompt tokens:  {stats['avg_prompt_tokens']:.1f}")
        print(f"Avg output tokens:  {stats['avg_completion_tokens']:.1f}")
        print("=" * 80)

        # Save traces
        trace_file = out_dir / f"traces_ctx{ctx_len}_c{args.concurrency}.json"
        save_traces(traces, str(trace_file))
        print(f"\n✓ Traces saved: {trace_file}")

        # Save statistics
        stats_file = out_dir / f"stats_ctx{ctx_len}_c{args.concurrency}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"✓ Statistics saved: {stats_file}")

    # Save summary of all runs
    if len(all_results) > 1:
        summary_file = out_dir / "summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Summary saved: {summary_file}")

        # Print summary table
        print(f"\n{'='*80}")
        print("SUMMARY: Latency vs Context Length")
        print(f"{'='*80}")
        print(f"{'Context':>10} {'P50':>10} {'P95':>10} {'P99':>10} {'Throughput':>12}")
        print(f"{'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
        for res in all_results:
            print(f"{res['context_length']:>10} "
                  f"{res['latency_median']:>9.3f}s "
                  f"{res['latency_p95']:>9.3f}s "
                  f"{res['latency_p99']:>9.3f}s "
                  f"{res['throughput_req_per_sec']:>11.2f} r/s")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
