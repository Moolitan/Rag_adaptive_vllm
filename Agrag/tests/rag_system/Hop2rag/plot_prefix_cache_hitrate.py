"""
绘制前缀缓存命中率随请求变化的图表

从 VLLMMonitor 生成的 CSV 中读取数据，绘制命中率曲线
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def detect_requests(df: pd.DataFrame, threshold: float = 100) -> pd.DataFrame:
    """
    检测请求边界，为每个请求分配 ID

    当 delta_prompt > threshold 时，认为是新请求的开始

    Args:
        df: VLLMMonitor 生成的 DataFrame
        threshold: 判断新请求的阈值（tokens）

    Returns:
        添加了 request_id 列的 DataFrame
    """
    request_id = 0
    request_ids = []

    for i, row in df.iterrows():
        delta_prompt = row.get('delta_prompt', 0)
        if delta_prompt > threshold:
            request_id += 1
        request_ids.append(request_id)

    df['request_id'] = request_ids
    return df


def aggregate_by_request(df: pd.DataFrame) -> pd.DataFrame:
    """
    按请求聚合数据

    Args:
        df: 包含 request_id 的 DataFrame

    Returns:
        每行代表一个请求的 DataFrame
    """
    # 过滤掉 request_id = 0（初始化阶段）
    df = df[df['request_id'] > 0].copy()

    if len(df) == 0:
        return pd.DataFrame()

    # 按 request_id 分组，取最后一个值（累积命中率）
    grouped = df.groupby('request_id').agg({
        'prefix_cache_hitrate_cumulative': 'last',
        'prefix_cache_queries_total': 'last',
        'prefix_cache_hits_total': 'last',
    }).reset_index()

    return grouped


def plot_hitrate(df: pd.DataFrame, output_path: str, title: str = "Prefix Cache Hit Rate"):
    """
    绘制命中率曲线

    Args:
        df: 聚合后的 DataFrame，包含 request_id 和 prefix_cache_hitrate_cumulative
        output_path: 输出图片路径
        title: 图表标题
    """
    if len(df) == 0:
        print("⚠️  没有数据可绘制")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制命中率曲线
    ax.plot(df['request_id'], df['prefix_cache_hitrate_cumulative'],
            marker='o', linewidth=2, markersize=4, label='Cumulative Hit Rate')

    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')

    # 设置标签
    ax.set_xlabel('Request ID', fontsize=12)
    ax.set_ylabel('Hit Rate (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # 设置 y 轴范围
    ax.set_ylim(0, 100)

    # 添加图例
    ax.legend(loc='lower right', fontsize=10)

    # 添加统计信息
    final_hitrate = df['prefix_cache_hitrate_cumulative'].iloc[-1]
    total_queries = df['prefix_cache_queries_total'].iloc[-1]
    total_hits = df['prefix_cache_hits_total'].iloc[-1]

    stats_text = f"Final Hit Rate: {final_hitrate:.2f}%\n"
    stats_text += f"Total Queries: {int(total_queries)} tokens\n"
    stats_text += f"Total Hits: {int(total_hits)} tokens"

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 保存图片
    plt.tight_layout()
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 图表已保存到: {output_file}")

    plt.close()


def plot_cumulative_hitrate(df: pd.DataFrame, output_path: str):
    """绘制累积命中率图"""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df['request_id'], df['prefix_cache_hitrate_cumulative'],
            marker='o', linewidth=2.5, markersize=5, color='#2E86AB',
            markerfacecolor='white', markeredgewidth=2)

    ax.set_xlabel('Request ID', fontsize=14, fontweight='bold')
    ax.set_ylabel('Hit Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Prefix Cache Cumulative Hit Rate', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax.set_ylim(0, 105)

    # 添加最终命中率标注
    final_hitrate = df['prefix_cache_hitrate_cumulative'].iloc[-1]
    ax.axhline(y=final_hitrate, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.text(df['request_id'].max() * 0.98, final_hitrate + 2,
            f'Final: {final_hitrate:.2f}%',
            ha='right', fontsize=12, color='red', fontweight='bold')

    # 保存图片
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 图表已保存到: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="绘制前缀缓存命中率图表",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法
  python plot_prefix_cache_hitrate.py \\
      --input tests/results/hop2rag_performance/vllm_metrics.csv \\
      --output tests/results/hop2rag_performance/plots/prefix_cache_hitrate.png

  # 自定义请求检测阈值
  python plot_prefix_cache_hitrate.py \\
      --input vllm_metrics.csv \\
      --output plots/hitrate.png \\
      --threshold 200
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='输入 CSV 文件路径（VLLMMonitor 生成的）'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='输出图片文件路径（PNG 格式）'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=100,
        help='判断新请求的阈值（delta_prompt > threshold）(默认: 100)'
    )

    args = parser.parse_args()

    # 读取 CSV
    print(f"📖 读取文件: {args.input}")
    df = pd.read_csv(args.input)
    print(f"   总记录数: {len(df)}")

    # 检查必要的列
    required_cols = ['delta_prompt', 'prefix_cache_hitrate_cumulative']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ 错误: CSV 缺少必要的列: {missing_cols}")
        print(f"   可用的列: {list(df.columns)}")
        return

    # 检测请求边界
    print(f"🔍 检测请求边界 (threshold={args.threshold})...")
    df = detect_requests(df, threshold=args.threshold)
    print(f"   检测到 {df['request_id'].max()} 个请求")

    # 按请求聚合
    print("📊 按请求聚合数据...")
    df_agg = aggregate_by_request(df)
    print(f"   聚合后记录数: {len(df_agg)}")

    if len(df_agg) == 0:
        print("⚠️  没有有效的请求数据")
        return

    # 绘图
    plot_cumulative_hitrate(df_agg, args.output)

    print("\n✅ 完成！")


if __name__ == '__main__':
    main()
