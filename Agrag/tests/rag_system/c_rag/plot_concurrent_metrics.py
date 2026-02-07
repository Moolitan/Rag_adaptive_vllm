"""
绘制并发性能测试的多个指标对比图

对于相同的 limit，比较不同并发数（max_workers）下的性能指标：
1. Prefix Cache Hit Rate (累积命中率)
2. Avg Prompt Throughput (平均提示词吞吐量)
3. Avg Generation Throughput (平均生成吞吐量)
4. GPU KV Cache Usage (GPU KV 缓存使用率)
5. Running/Waiting Requests (运行中/等待中的请求数)
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict
import glob


def find_csv_files(results_dir: Path, limit: int) -> List[Path]:
    """
    查找指定 limit 的所有 CSV 文件

    Args:
        results_dir: 结果目录
        limit: 问题数量限制

    Returns:
        CSV 文件路径列表，按并发数排序
    """
    pattern = str(results_dir / f"vllm_metrics_concurrent_l{limit}_c*.csv")
    files = glob.glob(pattern)

    if not files:
        return []

    # 提取并发数并排序
    file_info = []
    for f in files:
        path = Path(f)
        # 从文件名提取并发数: vllm_metrics_concurrent_l{limit}_c{workers}.csv
        # 使用正则表达式提取 workers 数字
        import re
        match = re.search(rf'_l{limit}_c(\d+)\.csv$', path.name)
        if match:
            try:
                workers = int(match.group(1))
                file_info.append((workers, path))
            except ValueError:
                continue

    # 按并发数排序
    file_info.sort(key=lambda x: x[0])
    return [path for _, path in file_info]


def load_csv_data(csv_files: List[Path]) -> Dict[int, pd.DataFrame]:
    """
    加载所有 CSV 文件

    Args:
        csv_files: CSV 文件路径列表

    Returns:
        {并发数: DataFrame} 字典
    """
    import re
    data = {}
    for csv_file in csv_files:
        # 从文件名提取并发数: vllm_metrics_concurrent_l{limit}_c{workers}.csv
        match = re.search(r'_l\d+_c(\d+)\.csv$', csv_file.name)
        if match:
            try:
                workers = int(match.group(1))
                df = pd.read_csv(csv_file)
                data[workers] = df
                print(f"  ✓ 加载 {csv_file.name}: {len(df)} 条记录")
            except Exception as e:
                print(f"  ✗ 加载失败 {csv_file.name}: {e}")

    return data


def plot_prefix_cache_hitrate(data: Dict[int, pd.DataFrame], output_path: Path, limit: int):
    """
    绘制不同并发数下的 Prefix Cache 累积命中率对比图
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, len(data)))

    for idx, (workers, df) in enumerate(sorted(data.items())):
        if 'prefix_cache_hitrate_cumulative' not in df.columns:
            continue

        # 过滤有效数据
        df_plot = df.copy()

        # 绘制曲线
        ax.plot(
            range(len(df_plot)),
            df_plot['prefix_cache_hitrate_cumulative'],
            marker='o',
            linewidth=2.5,
            markersize=4,
            color=colors[idx],
            label=f'{workers} workers',
            alpha=0.8
        )

        # 标注最终命中率
        final_hitrate = df_plot['prefix_cache_hitrate_cumulative'].iloc[-1]
        ax.annotate(
            f'{final_hitrate:.2f}%',
            xy=(len(df_plot) - 1, final_hitrate),
            xytext=(5, 0),
            textcoords='offset points',
            ha='left',
            va='center',
            fontsize=9,
            color=colors[idx],
            fontweight='bold'
        )

    ax.set_xlabel('Time (sampling points)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cumulative Hit Rate (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'Prefix Cache Hit Rate Comparison (limit={limit})',
                 fontsize=15, fontweight='bold', pad=20)

    ax.set_ylim(0, max(15, ax.get_ylim()[1]))
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax.set_axisbelow(True)
    ax.legend(loc='best', framealpha=0.95, edgecolor='gray')

    plt.tight_layout()
    output_file = output_path / f"prefix_cache_hitrate_l{limit}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ 保存: {output_file}")
    plt.close()


def plot_avg_prompt_throughput(data: Dict[int, pd.DataFrame], output_path: Path, limit: int):
    """
    绘制不同并发数下的平均提示词吞吐量对比图
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, len(data)))

    for idx, (workers, df) in enumerate(sorted(data.items())):
        # 使用 TokenRateAnalyzer 计算的吞吐量
        if 'prompt_toks_per_s' not in df.columns:
            continue

        df_plot = df[df['prompt_toks_per_s'] > 0].copy()

        if len(df_plot) == 0:
            continue

        ax.plot(
            range(len(df_plot)),
            df_plot['prompt_toks_per_s'],
            linewidth=2.5,
            color=colors[idx],
            label=f'{workers} workers',
            alpha=0.8
        )

        # 标注平均值
        mean_tput = df_plot['prompt_toks_per_s'].mean()
        ax.axhline(
            y=mean_tput,
            color=colors[idx],
            linestyle='--',
            linewidth=1.5,
            alpha=0.4
        )

    ax.set_xlabel('Time (sampling points)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Throughput (tokens/s)', fontsize=13, fontweight='bold')
    ax.set_title(f'Avg Prompt Throughput Comparison (limit={limit})',
                 fontsize=15, fontweight='bold', pad=20)

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax.set_axisbelow(True)
    ax.legend(loc='best', framealpha=0.95, edgecolor='gray')

    plt.tight_layout()
    output_file = output_path / f"avg_prompt_throughput_l{limit}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ 保存: {output_file}")
    plt.close()


def plot_avg_generation_throughput(data: Dict[int, pd.DataFrame], output_path: Path, limit: int):
    """
    绘制不同并发数下的平均生成吞吐量对比图
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, len(data)))

    for idx, (workers, df) in enumerate(sorted(data.items())):
        # 使用 TokenRateAnalyzer 计算的吞吐量
        if 'gen_toks_per_s' not in df.columns:
            continue

        df_plot = df[df['gen_toks_per_s'] > 0].copy()

        if len(df_plot) == 0:
            continue

        ax.plot(
            range(len(df_plot)),
            df_plot['gen_toks_per_s'],
            linewidth=2.5,
            color=colors[idx],
            label=f'{workers} workers',
            alpha=0.8
        )

        # 标注平均值
        mean_tput = df_plot['gen_toks_per_s'].mean()
        ax.axhline(
            y=mean_tput,
            color=colors[idx],
            linestyle='--',
            linewidth=1.5,
            alpha=0.4
        )

    ax.set_xlabel('Time (sampling points)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Throughput (tokens/s)', fontsize=13, fontweight='bold')
    ax.set_title(f'Avg Generation Throughput Comparison (limit={limit})',
                 fontsize=15, fontweight='bold', pad=20)

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax.set_axisbelow(True)
    ax.legend(loc='best', framealpha=0.95, edgecolor='gray')

    plt.tight_layout()
    output_file = output_path / f"avg_generation_throughput_l{limit}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ 保存: {output_file}")
    plt.close()


def plot_gpu_cache_usage(data: Dict[int, pd.DataFrame], output_path: Path, limit: int):
    """
    绘制不同并发数下的 GPU KV Cache 使用率对比图
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, len(data)))

    for idx, (workers, df) in enumerate(sorted(data.items())):
        if 'gpu_cache_raw' not in df.columns:
            continue

        # 转换为百分比
        df_plot = df.copy()
        df_plot['gpu_cache_pct'] = df_plot['gpu_cache_raw'] * 100.0

        ax.plot(
            range(len(df_plot)),
            df_plot['gpu_cache_pct'],
            linewidth=2.5,
            color=colors[idx],
            label=f'{workers} workers',
            alpha=0.8
        )

        # 标注最大值
        max_usage = df_plot['gpu_cache_pct'].max()
        max_idx = df_plot['gpu_cache_pct'].idxmax()
        ax.annotate(
            f'Max: {max_usage:.1f}%',
            xy=(max_idx, max_usage),
            xytext=(0, 10),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=9,
            color=colors[idx],
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor=colors[idx])
        )

    ax.set_xlabel('Time (sampling points)', fontsize=13, fontweight='bold')
    ax.set_ylabel('GPU KV Cache Usage (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'GPU KV Cache Usage Comparison (limit={limit})',
                 fontsize=15, fontweight='bold', pad=20)

    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax.set_axisbelow(True)
    ax.legend(loc='best', framealpha=0.95, edgecolor='gray')

    # 添加警戒线
    ax.axhline(y=80, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Warning (80%)')
    ax.axhline(y=90, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Critical (90%)')

    plt.tight_layout()
    output_file = output_path / f"gpu_cache_usage_l{limit}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ 保存: {output_file}")
    plt.close()


def plot_running_waiting_requests(data: Dict[int, pd.DataFrame], output_path: Path, limit: int):
    """
    绘制不同并发数下的运行中/等待中请求数对比图
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    colors = plt.cm.tab10(np.linspace(0, 1, len(data)))

    # 子图1: Running Requests
    for idx, (workers, df) in enumerate(sorted(data.items())):
        if 'running_raw' not in df.columns:
            continue

        ax1.plot(
            range(len(df)),
            df['running_raw'],
            linewidth=2.5,
            color=colors[idx],
            label=f'{workers} workers',
            alpha=0.8
        )

    ax1.set_ylabel('Running Requests', fontsize=13, fontweight='bold')
    ax1.set_title(f'Running Requests Comparison (limit={limit})',
                  fontsize=15, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax1.set_axisbelow(True)
    ax1.legend(loc='best', framealpha=0.95, edgecolor='gray')

    # 子图2: Waiting Requests
    for idx, (workers, df) in enumerate(sorted(data.items())):
        if 'waiting_raw' not in df.columns:
            continue

        ax2.plot(
            range(len(df)),
            df['waiting_raw'],
            linewidth=2.5,
            color=colors[idx],
            label=f'{workers} workers',
            alpha=0.8
        )

    ax2.set_xlabel('Time (sampling points)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Waiting Requests', fontsize=13, fontweight='bold')
    ax2.set_title(f'Waiting Requests Comparison (limit={limit})',
                  fontsize=15, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax2.set_axisbelow(True)
    ax2.legend(loc='best', framealpha=0.95, edgecolor='gray')

    plt.tight_layout()
    output_file = output_path / f"running_waiting_requests_l{limit}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ 保存: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="绘制并发性能测试的多个指标对比图",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 绘制 limit=20 的所有对比图
  python plot_concurrent_metrics.py \\
      --results-dir tests/results/hop2rag_performance_concurrent \\
      --limit 20 \\
      --output tests/results/hop2rag_performance_concurrent/plots

  # 绘制 limit=50 的所有对比图
  python plot_concurrent_metrics.py \\
      --results-dir tests/results/hop2rag_performance_concurrent \\
      --limit 50 \\
      --output tests/results/hop2rag_performance_concurrent/plots

输出文件 (对于 limit=20):
  - prefix_cache_hitrate_l20.png           (Prefix Cache 命中率对比)
  - avg_prompt_throughput_l20.png          (平均提示词吞吐量对比)
  - avg_generation_throughput_l20.png      (平均生成吞吐量对比)
  - gpu_cache_usage_l20.png                (GPU KV Cache 使用率对比)
  - running_waiting_requests_l20.png       (运行中/等待中请求数对比)
        """
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        required=True,
        help='结果目录路径（包含 vllm_metrics_concurrent_l*_c*.csv 文件）'
    )

    parser.add_argument(
        '--limit',
        type=int,
        required=True,
        help='问题数量限制（用于筛选对应的 CSV 文件）'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='输出图片目录路径'
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"绘制并发性能对比图 (limit={args.limit})")
    print(f"{'='*70}\n")

    # 查找 CSV 文件
    print(f"📂 查找 CSV 文件: {results_dir}")
    csv_files = find_csv_files(results_dir, args.limit)

    if not csv_files:
        print(f"❌ 错误: 未找到 limit={args.limit} 的 CSV 文件")
        print(f"   查找模式: vllm_metrics_concurrent_l{args.limit}_c*.csv")
        return

    print(f"   找到 {len(csv_files)} 个文件\n")

    # 加载数据
    print(f"📖 加载数据...")
    data = load_csv_data(csv_files)

    if not data:
        print(f"❌ 错误: 没有成功加载任何数据")
        return

    print(f"\n✓ 成功加载 {len(data)} 个并发配置的数据\n")

    # 绘制各个指标
    print(f"📊 绘制图表...\n")

    print(f"[1/5] Prefix Cache Hit Rate")
    plot_prefix_cache_hitrate(data, output_dir, args.limit)

    print(f"[2/5] Avg Prompt Throughput")
    plot_avg_prompt_throughput(data, output_dir, args.limit)

    print(f"[3/5] Avg Generation Throughput")
    plot_avg_generation_throughput(data, output_dir, args.limit)

    print(f"[4/5] GPU KV Cache Usage")
    plot_gpu_cache_usage(data, output_dir, args.limit)

    print(f"[5/5] Running/Waiting Requests")
    plot_running_waiting_requests(data, output_dir, args.limit)

    print(f"\n{'='*70}")
    print(f"✅ 完成！所有图表已保存到: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
