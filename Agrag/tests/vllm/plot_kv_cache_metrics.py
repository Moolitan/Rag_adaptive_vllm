"""
绘制 vLLM KV Cache 测试的性能指标图

基于 test_vllm_kv_cache.py 输出的 CSV 数据，绘制以下指标：
1. GPU KV Cache Usage (GPU KV 缓存使用率)
2. Prefix Cache Hit Rate (前缀缓存命中率)
3. Prompt Throughput (提示词吞吐量)
4. Generation Throughput (生成吞吐量)
5. Running/Waiting Requests (运行中/等待中的请求数)
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_csv_data(csv_path: Path) -> pd.DataFrame:
    """
    加载 CSV 文件

    Args:
        csv_path: CSV 文件路径

    Returns:
        DataFrame
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"  ✓ 加载 {csv_path.name}: {len(df)} 条记录")
        return df
    except Exception as e:
        print(f"  ✗ 加载失败 {csv_path.name}: {e}")
        return None


def plot_gpu_cache_usage(df: pd.DataFrame, output_path: Path):
    """
    绘制 GPU KV Cache 使用率图
    """
    if 'gpu_cache_raw' not in df.columns:
        print("  ⚠️ 跳过: 缺少 gpu_cache_raw 列")
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    # 转换为百分比
    df_plot = df.copy()
    df_plot['gpu_cache_pct'] = df_plot['gpu_cache_raw'] * 100.0

    ax.plot(
        range(len(df_plot)),
        df_plot['gpu_cache_pct'],
        linewidth=2.5,
        color='#2E86AB',
        alpha=0.8
    )

    # 标注最大值和平均值
    max_usage = df_plot['gpu_cache_pct'].max()
    mean_usage = df_plot['gpu_cache_pct'].mean()
    max_idx = df_plot['gpu_cache_pct'].idxmax()

    ax.annotate(
        f'Max: {max_usage:.1f}%',
        xy=(max_idx, max_usage),
        xytext=(0, 10),
        textcoords='offset points',
        ha='center',
        va='bottom',
        fontsize=10,
        color='#2E86AB',
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='#2E86AB')
    )

    # 平均值虚线
    ax.axhline(
        y=mean_usage,
        color='#2E86AB',
        linestyle='--',
        linewidth=1.5,
        alpha=0.4,
        label=f'Mean: {mean_usage:.1f}%'
    )

    ax.set_xlabel('Time (sampling points)', fontsize=13, fontweight='bold')
    ax.set_ylabel('GPU KV Cache Usage (%)', fontsize=13, fontweight='bold')
    ax.set_title('GPU KV Cache Usage Over Time',
                 fontsize=15, fontweight='bold', pad=20)

    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax.set_axisbelow(True)
    ax.legend(loc='best', framealpha=0.95, edgecolor='gray')

    # 添加警戒线
    ax.axhline(y=80, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Warning (80%)')
    ax.axhline(y=90, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Critical (90%)')

    plt.tight_layout()
    output_file = output_path / "gpu_cache_usage.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ 保存: {output_file}")
    plt.close()


def plot_prefix_cache_hitrate(df: pd.DataFrame, output_path: Path):
    """
    绘制 Prefix Cache 命中率图
    """
    if 'prefix_cache_hitrate_cumulative' not in df.columns:
        print("  ⚠️ 跳过: 缺少 prefix_cache_hitrate_cumulative 列")
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    df_plot = df.copy()

    ax.plot(
        range(len(df_plot)),
        df_plot['prefix_cache_hitrate_cumulative'],
        marker='o',
        linewidth=2.5,
        markersize=4,
        color='#A23B72',
        alpha=0.8
    )

    # 标注最终命中率
    final_hitrate = df_plot['prefix_cache_hitrate_cumulative'].iloc[-1]
    ax.annotate(
        f'Final: {final_hitrate:.2f}%',
        xy=(len(df_plot) - 1, final_hitrate),
        xytext=(5, 0),
        textcoords='offset points',
        ha='left',
        va='center',
        fontsize=10,
        color='#A23B72',
        fontweight='bold'
    )

    ax.set_xlabel('Time (sampling points)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cumulative Hit Rate (%)', fontsize=13, fontweight='bold')
    ax.set_title('Prefix Cache Hit Rate Over Time',
                 fontsize=15, fontweight='bold', pad=20)

    ax.set_ylim(0, max(15, ax.get_ylim()[1]))
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax.set_axisbelow(True)

    plt.tight_layout()
    output_file = output_path / "prefix_cache_hitrate.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ 保存: {output_file}")
    plt.close()


def plot_prompt_throughput(df: pd.DataFrame, output_path: Path):
    """
    绘制提示词吞吐量图
    """
    if 'prompt_toks_per_s' not in df.columns:
        print("  ⚠️ 跳过: 缺少 prompt_toks_per_s 列")
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    df_plot = df[df['prompt_toks_per_s'] > 0].copy()

    if len(df_plot) == 0:
        print("  ⚠️ 跳过: 没有有效的 prompt_toks_per_s 数据")
        return

    ax.plot(
        range(len(df_plot)),
        df_plot['prompt_toks_per_s'],
        linewidth=2.5,
        color='#F18F01',
        alpha=0.8
    )

    # 标注平均值
    mean_tput = df_plot['prompt_toks_per_s'].mean()
    ax.axhline(
        y=mean_tput,
        color='#F18F01',
        linestyle='--',
        linewidth=1.5,
        alpha=0.4,
        label=f'Mean: {mean_tput:.1f} tokens/s'
    )

    ax.set_xlabel('Time (sampling points)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Throughput (tokens/s)', fontsize=13, fontweight='bold')
    ax.set_title('Prompt Throughput Over Time',
                 fontsize=15, fontweight='bold', pad=20)

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax.set_axisbelow(True)
    ax.legend(loc='best', framealpha=0.95, edgecolor='gray')

    plt.tight_layout()
    output_file = output_path / "prompt_throughput.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ 保存: {output_file}")
    plt.close()


def plot_generation_throughput(df: pd.DataFrame, output_path: Path):
    """
    绘制生成吞吐量图
    """
    if 'gen_toks_per_s' not in df.columns:
        print("  ⚠️ 跳过: 缺少 gen_toks_per_s 列")
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    df_plot = df[df['gen_toks_per_s'] > 0].copy()

    if len(df_plot) == 0:
        print("  ⚠️ 跳过: 没有有效的 gen_toks_per_s 数据")
        return

    ax.plot(
        range(len(df_plot)),
        df_plot['gen_toks_per_s'],
        linewidth=2.5,
        color='#06A77D',
        alpha=0.8
    )

    # 标注平均值
    mean_tput = df_plot['gen_toks_per_s'].mean()
    ax.axhline(
        y=mean_tput,
        color='#06A77D',
        linestyle='--',
        linewidth=1.5,
        alpha=0.4,
        label=f'Mean: {mean_tput:.1f} tokens/s'
    )

    ax.set_xlabel('Time (sampling points)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Throughput (tokens/s)', fontsize=13, fontweight='bold')
    ax.set_title('Generation Throughput Over Time',
                 fontsize=15, fontweight='bold', pad=20)

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax.set_axisbelow(True)
    ax.legend(loc='best', framealpha=0.95, edgecolor='gray')

    plt.tight_layout()
    output_file = output_path / "generation_throughput.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ 保存: {output_file}")
    plt.close()


def plot_running_waiting_requests(df: pd.DataFrame, output_path: Path):
    """
    绘制运行中/等待中请求数图
    """
    if 'running_raw' not in df.columns or 'waiting_raw' not in df.columns:
        print("  ⚠️ 跳过: 缺少 running_raw 或 waiting_raw 列")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # 子图1: Running Requests
    ax1.plot(
        range(len(df)),
        df['running_raw'],
        linewidth=2.5,
        color='#5E60CE',
        alpha=0.8
    )

    # 标注平均值
    mean_running = df['running_raw'].mean()
    ax1.axhline(
        y=mean_running,
        color='#5E60CE',
        linestyle='--',
        linewidth=1.5,
        alpha=0.4,
        label=f'Mean: {mean_running:.1f}'
    )

    ax1.set_ylabel('Running Requests', fontsize=13, fontweight='bold')
    ax1.set_title('Running Requests Over Time',
                  fontsize=15, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax1.set_axisbelow(True)
    ax1.legend(loc='best', framealpha=0.95, edgecolor='gray')

    # 子图2: Waiting Requests
    ax2.plot(
        range(len(df)),
        df['waiting_raw'],
        linewidth=2.5,
        color='#E63946',
        alpha=0.8
    )

    # 标注平均值
    mean_waiting = df['waiting_raw'].mean()
    ax2.axhline(
        y=mean_waiting,
        color='#E63946',
        linestyle='--',
        linewidth=1.5,
        alpha=0.4,
        label=f'Mean: {mean_waiting:.1f}'
    )

    ax2.set_xlabel('Time (sampling points)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Waiting Requests', fontsize=13, fontweight='bold')
    ax2.set_title('Waiting Requests Over Time',
                  fontsize=15, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax2.set_axisbelow(True)
    ax2.legend(loc='best', framealpha=0.95, edgecolor='gray')

    plt.tight_layout()
    output_file = output_path / "running_waiting_requests.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ 保存: {output_file}")
    plt.close()


def plot_all_metrics_combined(df: pd.DataFrame, output_path: Path):
    """
    绘制所有关键指标的组合图
    """
    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    fig.suptitle('vLLM KV Cache Test - All Metrics', fontsize=16, fontweight='bold', y=0.995)

    # 1. GPU KV Cache Usage
    if 'gpu_cache_raw' in df.columns:
        ax = axes[0, 0]
        df_plot = df.copy()
        df_plot['gpu_cache_pct'] = df_plot['gpu_cache_raw'] * 100.0
        ax.plot(range(len(df_plot)), df_plot['gpu_cache_pct'], linewidth=2, color='#2E86AB')
        ax.set_ylabel('GPU Cache Usage (%)', fontweight='bold')
        ax.set_title('GPU KV Cache Usage', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)

    # 2. Prefix Cache Hit Rate
    if 'prefix_cache_hitrate_cumulative' in df.columns:
        ax = axes[0, 1]
        ax.plot(range(len(df)), df['prefix_cache_hitrate_cumulative'],
                linewidth=2, color='#A23B72', marker='o', markersize=3)
        ax.set_ylabel('Hit Rate (%)', fontweight='bold')
        ax.set_title('Prefix Cache Hit Rate', fontweight='bold')
        ax.grid(True, alpha=0.3)

    # 3. Prompt Throughput
    if 'prompt_toks_per_s' in df.columns:
        ax = axes[1, 0]
        df_plot = df[df['prompt_toks_per_s'] > 0]
        if len(df_plot) > 0:
            ax.plot(range(len(df_plot)), df_plot['prompt_toks_per_s'],
                    linewidth=2, color='#F18F01')
            ax.set_ylabel('Throughput (tokens/s)', fontweight='bold')
            ax.set_title('Prompt Throughput', fontweight='bold')
            ax.grid(True, alpha=0.3)

    # 4. Generation Throughput
    if 'gen_toks_per_s' in df.columns:
        ax = axes[1, 1]
        df_plot = df[df['gen_toks_per_s'] > 0]
        if len(df_plot) > 0:
            ax.plot(range(len(df_plot)), df_plot['gen_toks_per_s'],
                    linewidth=2, color='#06A77D')
            ax.set_ylabel('Throughput (tokens/s)', fontweight='bold')
            ax.set_title('Generation Throughput', fontweight='bold')
            ax.grid(True, alpha=0.3)

    # 5. Running Requests
    if 'running_raw' in df.columns:
        ax = axes[2, 0]
        ax.plot(range(len(df)), df['running_raw'], linewidth=2, color='#5E60CE')
        ax.set_xlabel('Time (sampling points)', fontweight='bold')
        ax.set_ylabel('Running Requests', fontweight='bold')
        ax.set_title('Running Requests', fontweight='bold')
        ax.grid(True, alpha=0.3)

    # 6. Waiting Requests
    if 'waiting_raw' in df.columns:
        ax = axes[2, 1]
        ax.plot(range(len(df)), df['waiting_raw'], linewidth=2, color='#E63946')
        ax.set_xlabel('Time (sampling points)', fontweight='bold')
        ax.set_ylabel('Waiting Requests', fontweight='bold')
        ax.set_title('Waiting Requests', fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_path / "all_metrics_combined.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ 保存: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="绘制 vLLM KV Cache 测试的性能指标图",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 绘制所有指标图
  python plot_kv_cache_metrics.py \\
      --csv tests/results/vllm_kv_cache/vllm_metrics.csv \\
      --output tests/results/vllm_kv_cache/plots

输出文件:
  - gpu_cache_usage.png              (GPU KV Cache 使用率)
  - prefix_cache_hitrate.png         (Prefix Cache 命中率)
  - prompt_throughput.png            (提示词吞吐量)
  - generation_throughput.png        (生成吞吐量)
  - running_waiting_requests.png     (运行中/等待中请求数)
  - all_metrics_combined.png         (所有指标组合图)
        """
    )

    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='CSV 文件路径（vllm_metrics.csv）'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='输出图片目录路径'
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"绘制 vLLM KV Cache 测试指标图")
    print(f"{'='*70}\n")

    # 加载数据
    print(f"📖 加载数据...")
    df = load_csv_data(csv_path)

    if df is None or len(df) == 0:
        print(f"❌ 错误: 无法加载数据或数据为空")
        return

    print(f"\n✓ 成功加载 {len(df)} 条记录\n")

    # 绘制各个指标
    print(f"📊 绘制图表...\n")

    print(f"[1/6] GPU KV Cache Usage")
    plot_gpu_cache_usage(df, output_dir)

    print(f"[2/6] Prefix Cache Hit Rate")
    plot_prefix_cache_hitrate(df, output_dir)

    print(f"[3/6] Prompt Throughput")
    plot_prompt_throughput(df, output_dir)

    print(f"[4/6] Generation Throughput")
    plot_generation_throughput(df, output_dir)

    print(f"[5/6] Running/Waiting Requests")
    plot_running_waiting_requests(df, output_dir)

    print(f"[6/6] All Metrics Combined")
    plot_all_metrics_combined(df, output_dir)

    print(f"\n{'='*70}")
    print(f"✅ 完成！所有图表已保存到: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
