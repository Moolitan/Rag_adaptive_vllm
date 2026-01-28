"""
分析 LLM call 的 prompt token 分布

用于确定 plot_prefix_cache_hitrate.py 中的 threshold 参数
生成 4 个独立的高质量图表
"""

import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

def load_llm_calls(json_path: str):
    """从 JSON 文件加载 LLM call 记录"""
    import json

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 支持两种格式
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "llm_calls" in data:
        return data["llm_calls"]
    else:
        raise ValueError(
            "Invalid JSON format. Expected a list or a dict with key 'llm_calls'"
        )


# =========================
# Global style (clean/report)
# =========================
def setup_style():
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "axes.titlepad": 12,
        "axes.linewidth": 1.0,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
    })

def _mkdir(output_path: str):
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def _percentile(x: np.ndarray, p: float) -> float:
    # numpy>=1.22: method; older: interpolation
    try:
        return float(np.percentile(x, p, method="linear"))
    except TypeError:
        return float(np.percentile(x, p, interpolation="linear"))

def analyze_distribution(token_counts: List[int]) -> Dict[str, Any]:
    if not token_counts:
        return {}

    x = np.asarray(token_counts, dtype=float)
    stats = {
        "total_calls": int(x.size),
        "min": int(np.min(x)),
        "max": int(np.max(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "percentiles": {
            "p10": int(round(_percentile(x, 10))),
            "p25": int(round(_percentile(x, 25))),
            "p50": int(round(_percentile(x, 50))),
            "p75": int(round(_percentile(x, 75))),
            "p90": int(round(_percentile(x, 90))),
            "p95": int(round(_percentile(x, 95))),
            "p99": int(round(_percentile(x, 99))),
        }
    }
    return stats

def suggest_threshold(token_counts: List[int]) -> Dict[str, Any]:
    if not token_counts:
        return {"threshold": 100, "reason": "No data available, using default", "stats": {}}

    stats = analyze_distribution(token_counts)

    # 你的策略：P10
    threshold = stats["percentiles"]["p10"]
    reason = f"Using P10 ({threshold}) as threshold"

    # 兜底逻辑（更稳一点）
    if threshold < 50:
        threshold = stats["percentiles"]["p25"]
        reason = f"Using P25 ({threshold}) as threshold (P10 too small)"
    if threshold < 50:
        threshold = 100
        reason = "Using default threshold 100 (distribution too skewed)"

    return {"threshold": int(threshold), "reason": reason, "stats": stats}

# -------------------------
# Helper: Freedman–Diaconis bins
# -------------------------
def _fd_bins(x: np.ndarray, min_bins: int = 12, max_bins: int = 60) -> int:
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return min_bins
    q25 = _percentile(x, 25)
    q75 = _percentile(x, 75)
    iqr = max(q75 - q25, 1e-9)
    bw = 2 * iqr / (x.size ** (1/3))
    if bw <= 0:
        return min_bins
    bins = int(np.ceil((x.max() - x.min()) / bw))
    return int(np.clip(bins, min_bins, max_bins))

# =========================
# Plot 1: Histogram
# =========================
def plot_histogram(token_counts: List[int], output_path: str, threshold: int = None):
    x = np.asarray(token_counts, dtype=float)
    stats = analyze_distribution(token_counts)

    fig, ax = plt.subplots(figsize=(11, 6.2), constrained_layout=True)

    bins = _fd_bins(x)
    ax.hist(x, bins=bins, edgecolor="white", linewidth=0.8, alpha=0.9)

    ax.set_title("Prompt Token Distribution (Histogram)")
    ax.set_xlabel("Prompt Tokens")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y")

    if threshold is not None:
        THRESHOLD_COLOR = "#f39c12"

        ax.axvline(
            threshold,
            linestyle="--",
            linewidth=2.4,
            color=THRESHOLD_COLOR,   # 👈 关键修改
            alpha=0.95,
            label=f"Threshold = {threshold}",
            zorder=3,
        )        
        # shading 更淡、更规整
        ax.axvspan(
            min(token_counts) - 5,
            threshold,
            color=THRESHOLD_COLOR,
            alpha=0.08,
            label="Below Threshold",
        )

        ax.axvspan(
            threshold,
            max(token_counts) + 5,
            color="tab:blue",
            alpha=0.05,
            label="Above Threshold",
        )

    # 统一信息框位置：右上角轴内
    info = (
        f"n = {stats['total_calls']}\n"
        f"mean = {stats['mean']:.1f}\n"
        f"std = {stats['std']:.1f}\n"
        f"median = {stats['median']:.1f}"
    )
    ax.text(
        0.98, 0.98, info, transform=ax.transAxes,
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="0.85")
    )

    if threshold is not None:
        ax.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="0.85")

    out = _mkdir(output_path)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"📊 Saved histogram: {out}")

# =========================
# Plot 2: Boxplot
# =========================
def plot_boxplot(token_counts: List[int], output_path: str, threshold: int = None):
    x = np.asarray(token_counts, dtype=float)
    stats = analyze_distribution(token_counts)

    fig, ax = plt.subplots(figsize=(7.6, 6.2), constrained_layout=True)

    ax.boxplot(
        x,
        vert=True,
        patch_artist=True,
        widths=0.45,
        boxprops=dict(alpha=0.85, linewidth=1.2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        medianprops=dict(linewidth=2.2),
        flierprops=dict(marker="o", markersize=4, alpha=0.35, markeredgewidth=0),
    )

    ax.set_title("Prompt Token Distribution (Boxplot)")
    ax.set_ylabel("Prompt Tokens")
    ax.set_xticks([1])
    ax.set_xticklabels(["All Calls"])
    ax.grid(True, axis="y")

    if threshold is not None:
        ax.axhline(threshold, linestyle="--", linewidth=2.0, label=f"Threshold = {threshold}")
        ax.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="0.85")

    # Quartile 标注固定在轴内右下角（规整，不漂）
    qtxt = (
        f"Q1 (P25) = {stats['percentiles']['p25']}\n"
        f"Q2 (P50) = {stats['percentiles']['p50']}\n"
        f"Q3 (P75) = {stats['percentiles']['p75']}"
    )
    ax.text(
        0.98, 0.06, qtxt, transform=ax.transAxes,
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="0.85"),
    )

    out = _mkdir(output_path)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"📊 Saved boxplot: {out}")

# =========================
# Plot 3: CDF
# =========================
def plot_cdf(token_counts: List[int], output_path: str, threshold: int = None):
    x = np.sort(np.asarray(token_counts, dtype=float))
    n = x.size
    cdf = np.arange(1, n + 1) / n

    fig, ax = plt.subplots(figsize=(11, 6.2), constrained_layout=True)
    ax.plot(x, cdf, linewidth=2.2, label="CDF")

    ax.set_title("Prompt Token Distribution (CDF)")
    ax.set_xlabel("Prompt Tokens")
    ax.set_ylabel("Cumulative Probability")
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(True)

    if threshold is not None:
        THRESHOLD_COLOR = "#f39c12"  # 橘黄色

        ax.axvline(
            threshold,
            linestyle="--",
            linewidth=2.4,
            color=THRESHOLD_COLOR,
            alpha=0.95,
            zorder=3,
        )

        p = float(np.searchsorted(x, threshold, side="right") / n)
        ax.scatter(
            [threshold],
            [p],
            s=70,
            color=THRESHOLD_COLOR,
            edgecolor="white",
            linewidth=1.2,
            zorder=5,
        )


        # 注释放轴内固定位置，更工整
        ax.text(
            0.02, 0.08,
            f"Threshold = {threshold}\nCDF = {p:.1%}",
            transform=ax.transAxes,
            ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="0.85"),
        )

    ax.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="0.85")

    out = _mkdir(output_path)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"📊 Saved CDF: {out}")

# =========================
# Plot 4: Stats Summary (clean table)
# =========================
def plot_statistics(token_counts: List[int], output_path: str, threshold: int = None):
    stats = analyze_distribution(token_counts)
    p = stats["percentiles"]

    rows = [
        ("Total calls", f"{stats['total_calls']}"),
        ("Min", f"{stats['min']}"),
        ("Max", f"{stats['max']}"),
        ("Mean", f"{stats['mean']:.1f}"),
        ("Median", f"{stats['median']:.1f}"),
        ("Std (ddof=1)", f"{stats['std']:.1f}"),
        ("", ""),
        ("P10", f"{p['p10']}"),
        ("P25 (Q1)", f"{p['p25']}"),
        ("P50 (Q2)", f"{p['p50']}"),
        ("P75 (Q3)", f"{p['p75']}"),
        ("P90", f"{p['p90']}"),
        ("P95", f"{p['p95']}"),
        ("P99", f"{p['p99']}"),
    ]
    if threshold is not None:
        rows += [("", ""), ("Suggested threshold", f"{threshold} tokens")]

    fig, ax = plt.subplots(figsize=(8.6, 7.2), constrained_layout=True)
    ax.axis("off")
    ax.text(
        0.5, 0.965,
        "Prompt Token Distribution (Summary)",
        ha="center",
        va="top",
        fontsize=16,
        fontweight="bold",
        transform=ax.transAxes,
    )
    cell_text = [[k, v] for k, v in rows]
    table = ax.table(
        cellText=cell_text,
        colLabels=["Metric", "Value"],
        cellLoc="left",
        colLoc="left",
        colWidths=[0.62, 0.38],
        loc="center",
    )
    ax.set_position([0.05, 0.01, 0.9, 0.9])

    # table styling: 行高一致 + 轻量条纹
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.55)

    # header
    for j in range(2):
        c = table[0, j]
        c.set_text_props(weight="bold", color="white")
        c.set_facecolor("0.25")
        c.set_edgecolor("white")

    # body
    nrows = len(rows)
    for i in range(1, nrows + 1):
        for j in range(2):
            c = table[i, j]
            c.set_edgecolor("0.85")
            c.set_linewidth(0.8)
            # stripe
            c.set_facecolor("0.98" if i % 2 == 0 else "1.0")
            # align value right for cleaner look
            if j == 1:
                c._loc = "right"

    # highlight threshold row if exists
    if threshold is not None:
        # last row index in table includes header row
        last_i = nrows
        for j in range(2):
            c = table[last_i, j]
            c.set_facecolor("0.93")
            c.set_text_props(weight="bold")

    out = _mkdir(output_path)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"📊 Saved summary: {out}")



def main():
    parser = argparse.ArgumentParser(
        description="分析 LLM call 的 prompt token 分布（生成 4 个独立图表）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 分析 LLM call 记录并生成 4 个图表
  python analyze_prompt_distribution.py \\
      --input tests/results/hop2rag_performance/llm_calls.json \\
      --output tests/results/hop2rag_performance/plots/prompt_dist \\
      --suggest-threshold

输出文件:
  - prompt_dist_histogram.png    (直方图)
  - prompt_dist_boxplot.png      (箱线图)
  - prompt_dist_cdf.png          (累积分布函数)
  - prompt_dist_statistics.png   (统计表)
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='输入 JSON 文件路径（包含 LLM call 记录）'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='输出文件路径前缀（不含扩展名）'
    )

    parser.add_argument(
        '--suggest-threshold',
        action='store_true',
        help='自动建议合适的 threshold 值'
    )

    parser.add_argument(
        '--threshold',
        type=int,
        default=None,
        help='手动指定 threshold 值（用于在图中标注）'
    )

    args = parser.parse_args()
    setup_style()
    # 读取 JSON
    print(f"📖 读取文件: {args.input}")
    llm_calls = load_llm_calls(args.input)
    print(f"   总 LLM calls: {len(llm_calls)}")

    if not llm_calls:
        print("❌ 错误: 没有 LLM call 记录")
        return

    # 提取 token 数量
    token_counts = [call.get("prompt_tokens", 0) for call in llm_calls]
    token_counts = [t for t in token_counts if t > 0]  # 过滤掉 0

    if not token_counts:
        print("❌ 错误: 没有有效的 token 数据")
        return

    print(f"   有效记录数: {len(token_counts)}")

    # 建议 threshold
    threshold = args.threshold
    if args.suggest_threshold or threshold is None:
        suggestion = suggest_threshold(token_counts)
        threshold = suggestion["threshold"]
        print(f"\n💡 建议的 threshold: {threshold}")
        print(f"   原因: {suggestion['reason']}")
        print(f"\n   统计信息:")
        print(f"   - 平均值: {suggestion['stats']['mean']:.1f}")
        print(f"   - 中位数: {suggestion['stats']['median']:.1f}")
        print(f"   - P10: {suggestion['stats']['percentiles']['p10']}")
        print(f"   - P25: {suggestion['stats']['percentiles']['p25']}")
        print(f"   - P90: {suggestion['stats']['percentiles']['p90']}")

    # 生成 4 个独立图表
    print(f"\n📊 生成图表...")
    output_prefix = args.output

    plot_histogram(token_counts, f"{output_prefix}_histogram.png", threshold)
    plot_boxplot(token_counts, f"{output_prefix}_boxplot.png", threshold)
    plot_cdf(token_counts, f"{output_prefix}_cdf.png", threshold)
    plot_statistics(token_counts, f"{output_prefix}_statistics.png", threshold)

    print("\n✅ 完成！")
    print(f"\n使用建议:")
    print(f"  python plot_prefix_cache_hitrate.py \\")
    print(f"      --input vllm_metrics.csv \\")
    print(f"      --output hitrate.png \\")
    print(f"      --threshold {threshold}")


if __name__ == '__main__':
    main()
