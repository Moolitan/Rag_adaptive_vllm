#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test vLLM baseline performance.

Runs a comprehensive baseline characterization and generates plots.

Usage:
    # Quick test (single context length)
    python tests/test_vllm_baseline.py --context-length 2000 --num-requests 50

    # Full sweep (multiple context lengths)
    python tests/test_vllm_baseline.py --context-sweep --num-requests 50

    # Test concurrency
    python tests/test_vllm_baseline.py --test-concurrency --num-requests 50
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # Agrag directory
TESTS_DIR = ROOT / "tests"
VLLM_BASELINE_DIR = Path(__file__).resolve().parent  # tests/vllm_baseline
RESULTS_DIR = TESTS_DIR / "results" / "vllm_baseline"
PLOTS_DIR = RESULTS_DIR / "plots"


def run_command(cmd, verbose=True):
    """Run a shell command"""
    if verbose:
        print(f"$ {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=not verbose, text=True)

    if result.returncode != 0:
        print(f"ERROR: Command failed with code {result.returncode}")
        if result.stderr:
            print(result.stderr)
        sys.exit(1)

    return result


def main():
    ap = argparse.ArgumentParser(description="Test vLLM baseline performance")

    # Test modes
    ap.add_argument(
        "--context-length",
        type=int,
        default=2000,
        help="Single context length to test (default: 2000)"
    )
    ap.add_argument(
        "--context-sweep",
        action="store_true",
        help="Sweep context lengths: 500, 1000, 2000, 4000"
    )
    ap.add_argument(
        "--test-concurrency",
        action="store_true",
        help="Test concurrency levels: 1, 2, 4, 8"
    )
    ap.add_argument(
        "--num-requests",
        type=int,
        default=50,
        help="Number of requests per test (default: 50)"
    )
    ap.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot generation"
    )

    args = ap.parse_args()

    print("=" * 80)
    print("vLLM Baseline Performance Test")
    print("=" * 80)
    print(f"Num requests: {args.num_requests}")
    print(f"Output: {RESULTS_DIR}")
    print("=" * 80)
    print()

    # Create directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Determine test mode
    if args.context_sweep:
        print("[MODE] Context length sweep: 500, 1000, 2000, 4000, 8000, 16000 tokens\n")
        context_arg = "--context-sweep"
        context_val = "500,1000,2000,4000,8000,16000"
    else:
        print(f"[MODE] Single context length: {args.context_length} tokens\n")
        context_arg = "--context-length"
        context_val = str(args.context_length)

    if args.test_concurrency:
        print("[MODE] Testing concurrency: 1, 2, 4, 8\n")
        concurrency_levels = [1, 2, 4, 8]
    else:
        concurrency_levels = [1]

    # Run benchmarks
    all_trace_files = []

    for concurrency in concurrency_levels:
        print(f"\n{'='*80}")
        print(f"Testing concurrency level: {concurrency}")
        print(f"{'='*80}\n")

        cmd = [
            sys.executable,
            str(VLLM_BASELINE_DIR / "vllm_baseline.py"),
            context_arg, context_val,
            "--num-requests", str(args.num_requests),
            "--concurrency", str(concurrency),
            "--out-dir", str(RESULTS_DIR / f"c{concurrency}"),
            "--verbose"
        ]

        run_command(cmd)

        # Collect trace files for plotting
        out_dir = RESULTS_DIR / f"c{concurrency}"
        if context_arg == "--context-sweep":
            for ctx in [500, 1000, 2000, 4000, 8000, 16000]:
                trace_file = out_dir / f"traces_ctx{ctx}_c{concurrency}.json"
                if trace_file.exists():
                    all_trace_files.append((ctx, concurrency, str(trace_file)))
        else:
            trace_file = out_dir / f"traces_ctx{args.context_length}_c{concurrency}.json"
            if trace_file.exists():
                all_trace_files.append((args.context_length, concurrency, str(trace_file)))

    # Generate plots
    if not args.skip_plots:
        print(f"\n{'='*80}")
        print("Generating Plots")
        print(f"{'='*80}\n")

        # If we have a summary file, use it
        if args.context_sweep or args.test_concurrency:
            # Combine all stats into a single summary
            import json
            all_stats = []

            for ctx, c, trace_file in all_trace_files:
                stats_file = Path(trace_file).parent / f"stats_ctx{ctx}_c{c}.json"
                if stats_file.exists():
                    with open(stats_file, 'r') as f:
                        stats = json.load(f)
                        all_stats.append(stats)

            if all_stats:
                summary_file = RESULTS_DIR / "summary_all.json"
                with open(summary_file, 'w') as f:
                    json.dump(all_stats, f, indent=2)

                print(f"Created summary: {summary_file}\n")

                # Generate plots from summary
                run_command([
                    sys.executable,
                    str(VLLM_BASELINE_DIR / "visualize_vllm_baseline.py"),
                    "--summary", str(summary_file),
                    "--out-dir", str(PLOTS_DIR)
                ])

        # Generate CDF for reference trace
        if all_trace_files:
            ref_trace = all_trace_files[0][2]  # First trace file
            run_command([
                sys.executable,
                str(VLLM_BASELINE_DIR / "visualize_vllm_baseline.py"),
                "--trace", ref_trace,
                "--out-dir", str(PLOTS_DIR)
            ])

    # Summary
    print(f"\n{'='*80}")
    print("Test Complete!")
    print(f"{'='*80}\n")

    print("Results saved to:")
    print(f"  Data:  {RESULTS_DIR}")
    if not args.skip_plots:
        print(f"  Plots: {PLOTS_DIR}")
    print()

    if not args.skip_plots:
        print("Generated plots:")
        for plot_file in sorted(PLOTS_DIR.glob("*.png")):
            print(f"  - {plot_file.name}")
        print()

    print("Next steps:")
    print(f"  1. View plots: cd {PLOTS_DIR} && ls *.png")
    print(f"  2. Check data: cat {RESULTS_DIR}/summary_all.json")
    print("  3. Compare with RAG benchmarks to identify overhead")
    print()


if __name__ == "__main__":
    main()
