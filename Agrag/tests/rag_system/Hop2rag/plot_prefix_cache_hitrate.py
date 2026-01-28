"""
绘制前缀缓存命中率时间序列图

直接使用 VLLMMonitor 采样的时间序列数据，不做请求聚合
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_cumulative_hitrate(df: pd.DataFrame, output_path: str):
    """
    绘制累积命中率时间序列图

    Args:
        df: VLLMMonitor 生成的 DataFrame
        output_path: 输出图片路径
    """
    if len(df) == 0:
        print("⚠️  没有数据可绘制")
        return

    # 过滤掉初始化阶段（命中率为 0 的点）和异常值（命中率 > 100%）
    # df_plot = df[
    #     (df['prefix_cache_hitrate_cumulative'] > 0) &
    #     (df['prefix_cache_hitrate_cumulative'] <= 100)
    # ].copy()

    df_plot = df.copy()

    if len(df_plot) == 0:
        print("⚠️  没有有效数据（所有命中率都为 0）")
        return

    fig, ax = plt.subplots(figsize=(12, 6))


    ax.plot( range(len(df_plot)), df_plot['prefix_cache_hitrate_cumulative'],
        marker='o', linewidth=2.5, markersize=5, color='#2E86AB',
        markerfacecolor='white', markeredgewidth=2)

    # 设置标签和标题
    ax.set_xlabel('Time (s)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Hit Rate (%)', fontsize=13, fontweight='bold')
    ax.set_title('Prefix Cache Cumulative Hit Rate (Time Series)',
                 fontsize=15, fontweight='bold', pad=20)

    # 设置 y 轴范围
    ax.set_ylim(0, 15)

    # 网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax.set_axisbelow(True)

    # 添加最终命中率水平线
    # 最终命中率
    final_hitrate = df_plot['prefix_cache_hitrate_cumulative'].iloc[-1]

    # ① 画红色水平虚线
    ax.axhline(
        y=final_hitrate,
        color='#e74c3c',
        linestyle='--',
        linewidth=2.5,
        alpha=0.6
    )

    # ② 在线旁边标注文字
    ax.annotate(
        f'Final: {final_hitrate:.2f}%',
        xy=(len(df_plot) - 1, final_hitrate),
        xytext=(-5, 10),
        textcoords='offset points',
        ha='right',
        va='bottom',
        fontsize=11,
        fontweight='bold',
        color='#e74c3c',
    )


    # 保存图片
    plt.tight_layout()
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 累积命中率图已保存到: {output_file}")
    plt.close()


def plot_delta_hitrate(df: pd.DataFrame, output_path: str):
    """
    绘制增量命中率时间序列图

    Args:
        df: VLLMMonitor 生成的 DataFrame
        output_path: 输出图片路径
    """
    if len(df) == 0:
        print("⚠️  没有数据可绘制")
        return

    # 过滤掉没有增量的点（delta_queries = 0）和异常值（命中率 > 100%）
    df_plot = df[
        (df['prefix_cache_delta_queries'] > 0) &
        (df['prefix_cache_hitrate_delta'] <= 100)
    ].copy()

    if len(df_plot) == 0:
        print("⚠️  没有有效数据（所有增量都为 0）")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # 使用不同颜色表示不同命中率范围
    colors = []
    for rate in df_plot['prefix_cache_hitrate_delta']:
        if rate >= 80:
            colors.append('#2ecc71')  # 绿色：高命中率
        elif rate >= 50:
            colors.append('#f39c12')  # 橙色：中等命中率
        else:
            colors.append('#e74c3c')  # 红色：低命中率

    # 绘制柱状图
    ax.bar(range(len(df_plot)), df_plot['prefix_cache_hitrate_delta'],
           color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)

    # 添加平均线
    mean_rate = df_plot['prefix_cache_hitrate_delta'].mean()
    ax.axhline(y=mean_rate, color='#34495e', linestyle='--',
               linewidth=2, alpha=0.7, label=f'Mean: {mean_rate:.2f}%')

    # 设置标签和标题
    ax.set_xlabel('Time (ms)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Hit Rate (%)', fontsize=13, fontweight='bold')
    ax.set_title('Prefix Cache Delta Hit Rate (Time Series)',
                 fontsize=15, fontweight='bold', pad=20)

    # 设置 y 轴范围
    ax.set_ylim(0, 105)

    # 网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1, axis='y')
    ax.set_axisbelow(True)


    # 添加颜色图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.8, label='High (≥80%)'),
        Patch(facecolor='#f39c12', alpha=0.8, label='Medium (50-80%)'),
        Patch(facecolor='#e74c3c', alpha=0.8, label='Low (<50%)'),
        plt.Line2D([0], [0], color='#34495e', linestyle='--',
                   linewidth=2, label=f'Mean: {mean_rate:.2f}%')
    ]
    ax.legend(handles=legend_elements, loc='upper right',
              framealpha=0.95, edgecolor='gray')

    # 保存图片
    plt.tight_layout()
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 增量命中率图已保存到: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="绘制前缀缓存命中率时间序列图",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法
  python plot_prefix_cache_hitrate.py \\
      --input tests/results/hop2rag_performance/vllm_metrics.csv \\
      --output tests/results/hop2rag_performance/plots/prefix_cache_hitrate.png

输出文件:
  - prefix_cache_hitrate.png        (累积命中率时间序列)
  - prefix_cache_hitrate_delta.png  (增量命中率时间序列)
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

    args = parser.parse_args()

    # 读取 CSV
    print(f"📖 读取文件: {args.input}")
    df = pd.read_csv(args.input)
    print(f"   总记录数: {len(df)}")

    # 检查必要的列
    required_cols = [
        'prefix_cache_hitrate_cumulative',
        'prefix_cache_hitrate_delta',
        'prefix_cache_queries_total',
        'prefix_cache_hits_total',
        'prefix_cache_delta_queries'
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ 错误: CSV 缺少必要的列: {missing_cols}")
        print(f"   可用的列: {list(df.columns)}")
        return

    # 绘制累积命中率图
    print(f"\n📊 绘制累积命中率图...")
    plot_cumulative_hitrate(df, args.output)

    # # 绘制增量命中率图
    # print(f"\n📊 绘制增量命中率图...")
    # delta_output = args.output.replace('.png', '_delta.png')
    # plot_delta_hitrate(df, delta_output)

    print("\n✅ 完成！")


if __name__ == '__main__':
    main()
