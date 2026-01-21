#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization for vLLM baseline benchmarks.

Generates publication-quality plots showing:
- Latency vs context length
- Latency vs concurrency
- Throughput scaling

Usage:
    python tests/visualize_vllm_baseline.py --summary tests/results/vllm_baseline/summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np

# Plotting imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Publication-quality settings
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.figsize': (10, 6),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2.5,
    'lines.markersize': 10,
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def plot_latency_vs_context(stats_list: list, output_file: str):
    """
    Plot latency percentiles vs context length.

    Shows how latency scales with input context size.
    Essential for identifying context-length regimes.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by context length
    stats_sorted = sorted(stats_list, key=lambda x: x['context_length'])

    contexts = [s['context_length'] for s in stats_sorted]
    p50 = [s['latency_median'] for s in stats_sorted]
    p95 = [s['latency_p95'] for s in stats_sorted]
    p99 = [s['latency_p99'] for s in stats_sorted]

    # Plot lines
    ax.plot(contexts, p50, marker='o', label='P50 (median)', linewidth=2.5, markersize=10)
    ax.plot(contexts, p95, marker='s', label='P95', linewidth=2.5, markersize=10)
    ax.plot(contexts, p99, marker='^', label='P99', linewidth=2.5, markersize=10)

    # Styling
    ax.set_xlabel('Context Length (tokens)')
    ax.set_ylabel('Latency (seconds)')
    ax.set_title('vLLM Baseline: Latency vs Context Length')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Use log scale if range is large
    if max(contexts) / min(contexts) > 10:
        ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    print(f"✓ Saved: {output_file}")


def plot_throughput_vs_context(stats_list: list, output_file: str):
    """
    Plot throughput vs context length.

    Shows how throughput degrades with larger contexts.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    stats_sorted = sorted(stats_list, key=lambda x: x['context_length'])

    contexts = [s['context_length'] for s in stats_sorted]
    throughput = [s['throughput_req_per_sec'] for s in stats_sorted]

    ax.plot(contexts, throughput, marker='o', linewidth=2.5, markersize=10,
            color=sns.color_palette("Set2")[2])

    # Styling
    ax.set_xlabel('Context Length (tokens)')
    ax.set_ylabel('Throughput (req/s)')
    ax.set_title('vLLM Baseline: Throughput vs Context Length')
    ax.grid(True, alpha=0.3)

    # Use log scale if range is large
    if max(contexts) / min(contexts) > 10:
        ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    print(f"✓ Saved: {output_file}")


def plot_latency_cdf(traces: list, output_file: str, context_length: int):
    """
    Plot CDF of latency for a single context length.

    Shows tail behavior.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    latencies = sorted([t['e2e_latency'] for t in traces if t['success']])
    n = len(latencies)

    if n == 0:
        print(f"⚠ No successful requests, skipping CDF plot")
        return

    cdf = np.arange(1, n + 1) / n
    ax.plot(latencies, cdf, linewidth=2.5, marker='o', markersize=4, markevery=n//10)

    # Mark percentiles
    p95_latency = latencies[int(n * 0.95)] if n > 0 else 0
    p99_latency = latencies[int(n * 0.99)] if n > 0 else 0

    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axhline(y=0.99, color='darkred', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axvline(x=p95_latency, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.axvline(x=p99_latency, color='darkred', linestyle=':', alpha=0.5, linewidth=1.5)

    # Annotations
    ax.text(p95_latency, 0.96, f'P95={p95_latency:.3f}s', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(p99_latency, 1.01, f'P99={p99_latency:.3f}s', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Styling
    ax.set_xlabel('Latency (seconds)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title(f'vLLM Baseline: Latency CDF (Context={context_length} tokens)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    print(f"✓ Saved: {output_file}")


def plot_concurrency_comparison(stats_by_concurrency: dict, output_file: str):
    """
    Compare latency across different concurrency levels.

    Shows scalability and contention effects.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    concurrencies = sorted(stats_by_concurrency.keys())
    p50 = [stats_by_concurrency[c]['latency_median'] for c in concurrencies]
    p95 = [stats_by_concurrency[c]['latency_p95'] for c in concurrencies]
    p99 = [stats_by_concurrency[c]['latency_p99'] for c in concurrencies]

    x = np.arange(len(concurrencies))
    width = 0.25

    colors = sns.color_palette("Set2", 3)
    ax.bar(x - width, p50, width, label='P50', color=colors[0], alpha=0.8, edgecolor='black')
    ax.bar(x, p95, width, label='P95', color=colors[1], alpha=0.8, edgecolor='black')
    ax.bar(x + width, p99, width, label='P99', color=colors[2], alpha=0.8, edgecolor='black')

    # Styling
    ax.set_xlabel('Concurrency Level')
    ax.set_ylabel('Latency (seconds)')
    ax.set_title('vLLM Baseline: Latency vs Concurrency')
    ax.set_xticks(x)
    ax.set_xticklabels(concurrencies)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    print(f"✓ Saved: {output_file}")


def main():
    ap = argparse.ArgumentParser(description="Visualize vLLM baseline benchmarks")

    ap.add_argument(
        "--summary",
        type=str,
        help="Summary JSON file (from context sweep)"
    )
    ap.add_argument(
        "--trace",
        type=str,
        help="Single trace file (for CDF)"
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "tests" / "results" / "vllm_baseline" / "plots"),
        help="Output directory for plots"
    )

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating vLLM baseline plots...\n")

    # If summary provided, generate scaling plots
    if args.summary:
        with open(args.summary, 'r') as f:
            stats_list = json.load(f)

        print(f"Loaded {len(stats_list)} test results from summary\n")

        # Group by concurrency level
        stats_by_concurrency = {}
        for stat in stats_list:
            c = stat.get('concurrency', 1)
            if c not in stats_by_concurrency:
                stats_by_concurrency[c] = []
            stats_by_concurrency[c].append(stat)

        # Plot latency vs context (for each concurrency level)
        for c, stats in stats_by_concurrency.items():
            plot_latency_vs_context(
                stats,
                str(out_dir / f'latency_vs_context_c{c}.png')
            )
            plot_throughput_vs_context(
                stats,
                str(out_dir / f'throughput_vs_context_c{c}.png')
            )

        # If multiple concurrency levels, compare them
        if len(stats_by_concurrency) > 1:
            # Get stats for a reference context length
            ref_context = stats_list[0]['context_length']
            stats_by_c = {}
            for c in stats_by_concurrency.keys():
                # Find stat with ref_context
                for stat in stats_by_concurrency[c]:
                    if stat['context_length'] == ref_context:
                        stats_by_c[c] = stat
                        break

            if len(stats_by_c) > 1:
                plot_concurrency_comparison(
                    stats_by_c,
                    str(out_dir / f'latency_vs_concurrency_ctx{ref_context}.png')
                )

    # If trace provided, generate CDF
    if args.trace:
        with open(args.trace, 'r') as f:
            traces = json.load(f)

        print(f"Loaded {len(traces)} traces\n")

        context_length = traces[0]['context_length'] if traces else 0
        plot_latency_cdf(
            traces,
            str(out_dir / f'latency_cdf_ctx{context_length}.png'),
            context_length
        )

    print(f"\n✓ All plots saved to: {out_dir}\n")


if __name__ == "__main__":
    main()
