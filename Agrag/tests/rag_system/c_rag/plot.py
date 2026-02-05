#!/usr/bin/env python3
"""
绘制单个问题中各LLM调用提示词之间的关联度和相似度热力图。

读取 performance_results.json，提取 llm_call_details 中的 prompt，
计算各 prompt 之间的相似度，并生成热力图。
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
matplotlib.rcParams['axes.unicode_minus'] = False


def load_performance_data(json_path: str) -> list:
    """加载性能测试结果数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_llm_prompts(question_data: dict) -> list[tuple[str, str]]:
    """
    从单个问题数据中提取所有LLM调用的节点名称和提示词
    
    返回: [(node_name, prompt), ...]
    """
    llm_calls = question_data.get('llm_call_details', [])
    prompts = []
    for call in llm_calls:
        node_name = call.get('node_name', 'unknown')
        prompt = call.get('prompt', '')
        if prompt:
            prompts.append((node_name, prompt))
    return prompts


def calculate_prompt_similarity(prompts: list[str]) -> np.ndarray:
    """
    使用TF-IDF和余弦相似度计算提示词之间的相似度矩阵
    """
    if len(prompts) < 2:
        return np.array([[1.0]])
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(prompts)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix


def plot_prompt_similarity_heatmap(
    similarity_matrix: np.ndarray,
    labels: list[str],
    question: str,
    save_path: str = None
):
    """
    绘制提示词相似度热力图
    """
    n = len(labels)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制热力图
    im = ax.imshow(similarity_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Cosine Similarity', rotation=-90, va="bottom", fontsize=12)
    
    # 设置坐标轴刻度
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    
    # 创建简短的标签（节点名称 + 序号）
    short_labels = [f"{i+1}. {label}" for i, label in enumerate(labels)]
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(short_labels, fontsize=10)
    
    # 在每个格子中添加数值
    for i in range(n):
        for j in range(n):
            value = similarity_matrix[i, j]
            text_color = "white" if value > 0.5 else "black"
            ax.text(j, i, f'{value:.2f}', ha="center", va="center", 
                   color=text_color, fontsize=9)
    
    # 设置标题和标签
    # 截断问题文本以适应标题
    short_question = question[:80] + "..." if len(question) > 80 else question
    ax.set_title(f'LLM Prompts Similarity Heatmap\nQuestion: "{short_question}"', 
                fontsize=14, pad=20)
    ax.set_xlabel('LLM Call (Node Name)', fontsize=12)
    ax.set_ylabel('LLM Call (Node Name)', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    
    plt.show()
    return fig


def plot_single_question_prompt_analysis(json_path: str, question_index: int = 0, save_path: str = None):
    """
    分析并绘制单个问题中各LLM调用提示词的相似度
    
    参数:
        json_path: performance_results.json 的路径
        question_index: 要分析的问题索引（从0开始）
        save_path: 图片保存路径（可选）
    """
    # 加载数据
    data = load_performance_data(json_path)
    
    if question_index >= len(data):
        print(f"错误: 问题索引 {question_index} 超出范围 (共 {len(data)} 个问题)")
        return
    
    question_data = data[question_index]
    question_text = question_data.get('question', 'Unknown Question')
    
    print(f"分析问题 #{question_index + 1}: {question_text}")
    print("-" * 60)
    
    # 提取提示词
    prompts_info = extract_llm_prompts(question_data)
    
    if len(prompts_info) < 2:
        print("警告: 该问题的LLM调用数量不足以进行相似度分析")
        return
    
    # 分离节点名称和提示词
    node_names = [info[0] for info in prompts_info]
    prompts = [info[1] for info in prompts_info]
    
    print(f"找到 {len(prompts)} 个LLM调用:")
    for i, (name, prompt) in enumerate(prompts_info):
        print(f"  {i+1}. {name}: {len(prompt)} 字符")
    print()
    
    # 计算相似度矩阵
    similarity_matrix = calculate_prompt_similarity(prompts)
    
    # 打印相似度统计
    upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
    if len(upper_triangle) > 0:
        print(f"相似度统计:")
        print(f"  平均相似度: {np.mean(upper_triangle):.4f}")
        print(f"  最大相似度: {np.max(upper_triangle):.4f}")
        print(f"  最小相似度: {np.min(upper_triangle):.4f}")
        print()
    
    # 绘制热力图
    plot_prompt_similarity_heatmap(
        similarity_matrix,
        node_names,
        question_text,
        save_path
    )
    
    return similarity_matrix, node_names


def main():
    """主函数"""
    # 配置路径
    json_path = "/home/wjj/Rag_adaptive_vllm/Agrag/tests/results/crag_performance/performance_results.json"
    output_dir = "/home/wjj/Rag_adaptive_vllm/Agrag/tests/results/crag_performance"
    
    # 分析第一个问题（可以修改 question_index 来分析其他问题）
    question_index = 0
    save_path = f"{output_dir}/prompt_similarity_q{question_index + 1}.png"
    
    plot_single_question_prompt_analysis(
        json_path=json_path,
        question_index=question_index,
        save_path=save_path
    )


if __name__ == "__main__":
    main()
