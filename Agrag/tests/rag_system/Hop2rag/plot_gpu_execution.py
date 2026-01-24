#!/usr/bin/env python3
"""
GPU Execution Timeline Visualization

绘制 LLM / Retriever / CPU 节点的时序图，展示执行时间分布。
"""

import json
import sys
from pathlib import Path

# Setup paths
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARN] matplotlib not found, visualization disabled")


# 颜色方案
LLM_COLOR = 'limegreen'       # 绿色
RETRIEVER_COLOR = 'mediumorchid'  # 紫色
CPU_COLOR = 'gold'            # 黄色


def load_performance_data(json_path: str) -> list:
    """加载性能数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def merge_consecutive_nodes(records: list) -> list:
    """
    合并相同 node_name 和 node_type 的连续记录

    对于在记录列表中连续出现的相同节点，合并为一个时间段（从最早开始到最晚结束）
    """
    if not records:
        return []

    # 首先按时间排序（确保按开始时间顺序处理）
    sorted_records = sorted(records, key=lambda x: x['timestamp'] - x.get('latency', 0))

    merged = []
    current_group = None

    for record in sorted_records:
        node_name = record.get('node_name', 'Unknown')
        node_type = record.get('node_type', 'cpu')

        start_time = record['timestamp'] - record['latency']
        end_time = record['timestamp']
        llm_latency = record.get('llm_latency', 0)
        retriever_latency = record.get('retriever_latency', 0)
        doc_count = record.get('doc_count', 0)

        # 如果是第一条记录，或者与上一组节点不同，开始新组
        if current_group is None or \
           current_group['node_name'] != node_name or \
           current_group['node_type'] != node_type:

            # 保存上一组（如果存在）
            if current_group is not None:
                latency = current_group['end_time'] - current_group['start_time']
                merged.append({
                    'node_name': current_group['node_name'],
                    'node_type': current_group['node_type'],
                    'timestamp': current_group['end_time'],
                    'latency': latency,
                    'llm_latency': current_group['llm_latency'],
                    'retriever_latency': current_group['retriever_latency'],
                    'doc_count': current_group['doc_count'],
                })

            # 开始新组
            current_group = {
                'node_name': node_name,
                'node_type': node_type,
                'start_time': start_time,
                'end_time': end_time,
                'llm_latency': llm_latency,
                'retriever_latency': retriever_latency,
                'doc_count': doc_count,
            }
        else:
            # 合并到当前组（连续的相同节点）
            current_group['start_time'] = min(current_group['start_time'], start_time)
            current_group['end_time'] = max(current_group['end_time'], end_time)
            current_group['llm_latency'] += llm_latency
            current_group['retriever_latency'] += retriever_latency
            current_group['doc_count'] = max(current_group['doc_count'], doc_count)

    # 保存最后一组
    if current_group is not None:
        latency = current_group['end_time'] - current_group['start_time']
        merged.append({
            'node_name': current_group['node_name'],
            'node_type': current_group['node_type'],
            'timestamp': current_group['end_time'],
            'latency': latency,
            'llm_latency': current_group['llm_latency'],
            'retriever_latency': current_group['retriever_latency'],
            'doc_count': current_group['doc_count'],
        })

    return merged


def extract_timeline_data(records: list) -> tuple:
    """
    提取时序数据（合并相同节点）

    Returns:
        tuple: (llm_nodes, retriever_nodes, cpu_nodes, base_timestamp)
    """
    if not records:
        return [], [], [], 0

    # 合并相同节点
    merged_records = merge_consecutive_nodes(records)

    # 找到最早的时间戳作为基准（时间0）
    start_times = [r['timestamp'] - r['latency'] for r in merged_records]
    base_timestamp = min(start_times) if start_times else 0

    llm_nodes = []
    retriever_nodes = []
    cpu_nodes = []

    for record in merged_records:
        start_time = record['timestamp'] - record['latency']
        relative_start_ms = (start_time - base_timestamp) * 1000
        duration_ms = record['latency'] * 1000
        node_name = record.get('node_name', 'Unknown')
        node_type = record.get('node_type', 'cpu')

        label = node_name[:12]

        if node_type == 'llm':
            llm_nodes.append((relative_start_ms, duration_ms, label))
        elif node_type == 'retriever':
            retriever_nodes.append((relative_start_ms, duration_ms, label))
        else:
            cpu_nodes.append((relative_start_ms, duration_ms, label))

    return llm_nodes, retriever_nodes, cpu_nodes, base_timestamp


def plot_execution_timeline(
    llm_nodes: list,
    retriever_nodes: list,
    cpu_nodes: list,
    output_path: str = None,
    title: str = "Hop2RAG Execution Timeline"
):
    """绘制执行时序图（甘特图风格）"""
    if not HAS_MATPLOTLIB:
        print("[WARN] matplotlib not available")
        return None, None

    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)

    # Y轴位置
    llm_y = 2
    retriever_y = 1
    cpu_y = 0
    bar_height = 0.6

    # 计算最大时间用于X轴范围
    all_times = llm_nodes + retriever_nodes + cpu_nodes
    if all_times:
        max_time = max(start + duration for start, duration, _ in all_times)
    else:
        max_time = 1000

    # 绘制 LLM nodes (绿色)
    for start, duration, label in llm_nodes:
        ax.barh(llm_y, duration, left=start, height=bar_height,
                color=LLM_COLOR, alpha=0.85, edgecolor='black', linewidth=0.8)
        if duration > max_time * 0.03:
            ax.text(start + duration / 2, llm_y, f'{duration:.0f}ms',
                   ha='center', va='center', fontsize=8, color='black', fontweight='bold')

    # 绘制 Retriever nodes (紫色)
    for start, duration, label in retriever_nodes:
        ax.barh(retriever_y, duration, left=start, height=bar_height,
                color=RETRIEVER_COLOR, alpha=0.85, edgecolor='black', linewidth=0.8)
        if duration > max_time * 0.03:
            ax.text(start + duration / 2, retriever_y, f'{duration:.0f}ms',
                   ha='center', va='center', fontsize=8, color='white', fontweight='bold')

    # 绘制 CPU nodes (黄色)
    for start, duration, label in cpu_nodes:
        ax.barh(cpu_y, duration, left=start, height=bar_height,
                color=CPU_COLOR, alpha=0.85, edgecolor='black', linewidth=0.8)

    # 设置Y轴
    ax.set_yticks([cpu_y, retriever_y, llm_y])
    ax.set_yticklabels(['CPU Node', 'Retriever', 'LLM Node'], fontsize=11)
    ax.set_ylim(-0.5, 2.8)

    # 设置X轴
    ax.set_xlim(-max_time * 0.02, max_time * 1.05)
    ax.set_xlabel('Time (ms)', fontsize=12)

    # 设置标题
    ax.set_title(title, fontsize=14, fontweight='bold')

    # 网格线
    ax.grid(True, alpha=0.3, axis='x')

    # 图例
    legend_patches = [
        mpatches.Patch(color=LLM_COLOR, label=f'LLM ({len(llm_nodes)} nodes)', alpha=0.85),
        mpatches.Patch(color=RETRIEVER_COLOR, label=f'Retriever ({len(retriever_nodes)} nodes)', alpha=0.85),
        mpatches.Patch(color=CPU_COLOR, label=f'CPU ({len(cpu_nodes)} nodes)', alpha=0.85),
    ]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=10)

    # 添加统计信息
    total_llm_time = sum(d for _, d, _ in llm_nodes) if llm_nodes else 0
    total_ret_time = sum(d for _, d, _ in retriever_nodes) if retriever_nodes else 0
    stats_text = f'LLM: {total_llm_time:.0f}ms | Retriever: {total_ret_time:.0f}ms'
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved timeline: {output_path}")

    return fig, ax


def plot_detailed_gantt(
    llm_nodes: list,
    retriever_nodes: list,
    cpu_nodes: list,
    output_path: str = None,
    title: str = "Hop2RAG Detailed Execution Gantt Chart"
):
    """绘制详细甘特图（每个节点一行）"""
    if not HAS_MATPLOTLIB:
        return None, None

    # 合并并按时间排序
    all_events = []
    for start, duration, label in llm_nodes:
        all_events.append(('llm', start, duration, label))
    for start, duration, label in retriever_nodes:
        all_events.append(('retriever', start, duration, label))
    for start, duration, label in cpu_nodes:
        all_events.append(('cpu', start, duration, label))

    all_events.sort(key=lambda x: x[1])  # 按开始时间排序

    if not all_events:
        return None, None

    fig, ax = plt.subplots(figsize=(14, max(6, len(all_events) * 0.35)), dpi=150)

    # 颜色
    color_map = {
        'llm': LLM_COLOR,
        'retriever': RETRIEVER_COLOR,
        'cpu': CPU_COLOR,
    }

    # 计算相对时间
    max_time = max(start + duration for _, start, duration, _ in all_events)

    # 绘制每个事件
    y_labels = []
    for i, (event_type, start, duration, label) in enumerate(all_events):
        color = color_map[event_type]

        ax.barh(i, duration, left=start, height=0.6,
                color=color, alpha=0.85, edgecolor='black', linewidth=0.8)

        # 添加时间标注（如果宽度足够）
        if duration > max_time * 0.02:
            text_color = 'black' if event_type in ['llm', 'cpu'] else 'white'
            ax.text(start + duration / 2, i, f'{duration:.0f}ms',
                   ha='center', va='center', fontsize=8, color=text_color, fontweight='bold')

        y_labels.append(f"{label} ({event_type})")

    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # 图例
    legend_patches = [
        mpatches.Patch(color=LLM_COLOR, label='LLM'),
        mpatches.Patch(color=RETRIEVER_COLOR, label='Retriever'),
        mpatches.Patch(color=CPU_COLOR, label='CPU'),
    ]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=10)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved detailed Gantt: {output_path}")

    return fig, ax


def plot_all_questions(data: list, output_dir: str = None):
    """为每个问题绘制时序图"""
    if not HAS_MATPLOTLIB:
        return
    gatt_dir = output_dir / "gantt"
    timeline_dir = output_dir / "timeline"

    gatt_dir.mkdir(parents=True, exist_ok=True)
    timeline_dir.mkdir(parents=True, exist_ok=True)


    for i, item in enumerate(data):
        question = item.get('question', f'Question {i+1}')
        records = item.get('records', [])

        short_question = question[:50] + '...' if len(question) > 50 else question

        llm_nodes, retriever_nodes, cpu_nodes, _ = extract_timeline_data(records)

        timeline_output_path = None
        if timeline_dir:
            timeline_output_path = timeline_dir / f'timeline_q{i+1}.png'

        fig, ax = plot_execution_timeline(
            llm_nodes,
            retriever_nodes,
            cpu_nodes,
            output_path=str(timeline_output_path) if timeline_output_path else None,
            title=f'Q{i+1}: {short_question}'
        )
        if fig:
            plt.close(fig)

        # 也绘制详细甘特图
        if gatt_dir:
            detail_path = gatt_dir / f'gantt_q{i+1}.png'
            fig2, ax2 = plot_detailed_gantt(
                llm_nodes,
                retriever_nodes,
                cpu_nodes,
                output_path=str(detail_path),
                title=f'Q{i+1} Detailed: {short_question}'
            )
            if fig2:
                plt.close(fig2)


def plot_combined_timeline(data: list, output_path: str = None):
    """绘制所有问题的合并时序图"""
    if not HAS_MATPLOTLIB:
        return None, None

    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)

    llm_y = 2
    retriever_y = 1
    cpu_y = 0
    bar_height = 0.6

    total_llm = 0
    total_retriever = 0
    total_cpu = 0
    total_llm_time = 0
    total_ret_time = 0
    max_time = 0

    for item in data:
        records = item.get('records', [])
        llm_nodes, retriever_nodes, cpu_nodes, _ = extract_timeline_data(records)

        total_llm += len(llm_nodes)
        total_retriever += len(retriever_nodes)
        total_cpu += len(cpu_nodes)

        for start, duration, _ in llm_nodes:
            ax.barh(llm_y, duration, left=start, height=bar_height,
                    color=LLM_COLOR, alpha=0.7, edgecolor='black', linewidth=0.5)
            total_llm_time += duration
            max_time = max(max_time, start + duration)

        for start, duration, _ in retriever_nodes:
            ax.barh(retriever_y, duration, left=start, height=bar_height,
                    color=RETRIEVER_COLOR, alpha=0.7, edgecolor='black', linewidth=0.5)
            total_ret_time += duration
            max_time = max(max_time, start + duration)

        for start, duration, _ in cpu_nodes:
            ax.barh(cpu_y, duration, left=start, height=bar_height,
                    color=CPU_COLOR, alpha=0.7, edgecolor='black', linewidth=0.5)
            max_time = max(max_time, start + duration)

    # 设置轴
    ax.set_yticks([cpu_y, retriever_y, llm_y])
    ax.set_yticklabels(['CPU Node', 'Retriever', 'LLM Node'], fontsize=11)
    ax.set_ylim(-0.5, 2.8)
    ax.set_xlim(-max_time * 0.02, max_time * 1.05)
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_title('Hop2RAG Combined Execution Timeline', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # 图例
    legend_patches = [
        mpatches.Patch(color=LLM_COLOR, label=f'LLM ({total_llm} nodes)', alpha=0.7),
        mpatches.Patch(color=RETRIEVER_COLOR, label=f'Retriever ({total_retriever} nodes)', alpha=0.7),
        mpatches.Patch(color=CPU_COLOR, label=f'CPU ({total_cpu} nodes)', alpha=0.7),
    ]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=10)

    # 统计
    stats = f'LLM: {total_llm_time:.0f}ms | Retriever: {total_ret_time:.0f}ms | Questions: {len(data)}'
    ax.text(0.02, 0.95, stats, transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved combined timeline: {output_path}")

    return fig, ax


def main():
    """主函数"""
    if not HAS_MATPLOTLIB:
        print("[ERROR] matplotlib is required")
        return

    # 路径配置
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent

    json_path = project_root / 'tests' / 'results' / 'hop2rag_performance' / 'performance_results.json'
    output_dir = project_root / 'tests' / 'results' / 'hop2rag_performance' / 'plots'

    print("=" * 60)
    print("Hop2RAG Execution Timeline Visualization")
    print("=" * 60)
    print(f"Data source: {json_path}")
    print(f"Output dir:  {output_dir}")
    print()

    # 加载数据
    data = load_performance_data(json_path)
    print(f"Loaded {len(data)} question(s)")

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 绘制每个问题的时序图
    print("\n--- Plotting individual timelines ---")
    plot_all_questions(data, output_dir=output_dir)

    # 绘制合并时序图
    print("\n--- Plotting combined timeline ---")
    fig, ax = plot_combined_timeline(data, output_path=output_dir / 'timeline_combined.png')
    if fig:
        plt.close(fig)

    print()
    print("=" * 60)
    print(f"All plots saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
