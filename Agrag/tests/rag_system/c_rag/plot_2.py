#!/usr/bin/env python3
"""
LLM Prompt Prefix Embedding Similarity Analysis

分析单个问题中各LLM调用的 prompt prefix 语义相似度：
1. Embedding-level Prefix Similarity (余弦相似度热力图)
2. 2D 降维可视化 (PCA)
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import List, Tuple, Dict
from pathlib import Path

# Embedding 相关
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

# Tokenizer (用于提取 prefix)
from transformers import AutoTokenizer

# ========== 超参数配置 ==========
# 最大 prefix token 数量
K_MAX_PREFIX_TOKENS = 128

# 模型路径 (本地 e5-large 模型)
E5_MODEL_PATH = "/mnt/Large_Language_Model_Lab_1/模型/rag_models/models--intfloat--e5-large/snapshots/4dc6d853a804b9c8886ede6dda8a073b7dc08a81"

# 输入 JSON 文件路径
INPUT_JSON = "/home/wjj/Rag_adaptive_vllm/Agrag/tests/results/crag_performance/performance_results.json"

# 输出图片目录
OUTPUT_DIR = "./figures"

# 要分析的问题索引 (从0开始)
QUESTION_INDEX = 0

# 设置 matplotlib 字体
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False


def load_json_data(json_path: str) -> List[Dict]:
    """加载 JSON 数据文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_llm_calls(question_data: Dict) -> List[Dict]:
    """从单个问题数据中提取 LLM 调用信息"""
    llm_call_details = question_data.get('llm_call_details', [])
    calls = []
    for idx, call in enumerate(llm_call_details):
        calls.append({
            'call_id': idx,
            'node_name': call.get('node_name', 'unknown'),
            'prompt_text': call.get('prompt', ''),
            'response_text': call.get('response', '')
        })
    return calls


def extract_prefix_text(
    prompt_text: str,
    tokenizer: AutoTokenizer,
    k: int = K_MAX_PREFIX_TOKENS
) -> str:
    """
    提取 prompt 的前 K 个 token 对应的文本
    """
    tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    actual_k = min(k, len(tokens))
    prefix_tokens = tokens[:actual_k]
    prefix_text = tokenizer.decode(prefix_tokens)
    return prefix_text


def compute_embedding_similarity(
    prefix_texts: List[str],
    model: SentenceTransformer
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 Embedding-level Prefix Similarity
    
    Returns:
        (相似度矩阵, embeddings)
    """
    embeddings = model.encode(prefix_texts, convert_to_numpy=True, show_progress_bar=True)
    sim_matrix = cosine_similarity(embeddings)
    return sim_matrix, embeddings


def plot_heatmap(
    matrix: np.ndarray,
    labels: List[str],
    title: str,
    save_path: str
):
    """绘制相似度热力图"""
    n = len(labels)
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Cosine Similarity', rotation=-90, va="bottom", fontsize=12)
    
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    
    short_labels = [f"{i+1}.{lbl[:15]}" for i, lbl in enumerate(labels)]
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short_labels, fontsize=9)
    
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f'{val:.2f}', ha="center", va="center", color=color, fontsize=8)
    
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel('LLM Call', fontsize=12)
    ax.set_ylabel('LLM Call', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_2d_embedding(
    embeddings: np.ndarray,
    labels: List[str],
    node_names: List[str],
    title: str,
    save_path: str
):
    """绘制 embedding 的 2D 降维图 (PCA)"""
    n_samples = embeddings.shape[0]
    if n_samples < 2:
        print("样本数量不足，无法进行 2D 降维")
        return
    
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    unique_nodes = list(set(node_names))
    color_map = plt.cm.get_cmap('tab10', len(unique_nodes))
    node_to_color = {node: color_map(i) for i, node in enumerate(unique_nodes)}
    colors = [node_to_color[name] for name in node_names]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    ax.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=colors,
        s=200,
        alpha=0.7,
        edgecolors='black',
        linewidths=1
    )
    
    for i, label in enumerate(labels):
        ax.annotate(
            label,
            (embeddings_2d[i, 0], embeddings_2d[i, 1]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            alpha=0.8
        )
    
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=node_to_color[node],
                   markersize=10, label=node)
        for node in unique_nodes
    ]
    ax.legend(handles=legend_handles, title='Node Type', loc='best', fontsize=9)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax.set_title(title, fontsize=14, pad=15)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("LLM Prompt Prefix Embedding Similarity Analysis")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载数据
    print(f"\n[1] Loading data from: {INPUT_JSON}")
    data = load_json_data(INPUT_JSON)
    print(f"    Total questions: {len(data)}")
    
    if QUESTION_INDEX >= len(data):
        print(f"Error: Question index {QUESTION_INDEX} out of range")
        return
    
    question_data = data[QUESTION_INDEX]
    question_text = question_data.get('question', 'Unknown')
    print(f"\n[2] Analyzing Question #{QUESTION_INDEX + 1}: \"{question_text[:60]}...\"")
    
    # 提取 LLM 调用
    llm_calls = extract_llm_calls(question_data)
    print(f"\n[3] Found {len(llm_calls)} LLM calls")
    
    if len(llm_calls) < 2:
        print("Error: Need at least 2 LLM calls for similarity analysis")
        return
    
    # 加载模型
    print(f"\n[4] Loading tokenizer and embedding model...")
    tokenizer = AutoTokenizer.from_pretrained(E5_MODEL_PATH)
    embed_model = SentenceTransformer(E5_MODEL_PATH)
    
    # 提取 prefix
    print(f"\n[5] Extracting prefix (K={K_MAX_PREFIX_TOKENS})")
    prefix_texts = []
    node_names = []
    labels = []
    
    for call in llm_calls:
        prefix = extract_prefix_text(call['prompt_text'], tokenizer, K_MAX_PREFIX_TOKENS)
        prefix_texts.append(prefix)
        node_names.append(call['node_name'])
        labels.append(f"{call['call_id']+1}.{call['node_name']}")
        print(f"    Call {call['call_id']+1} ({call['node_name']}): {len(prefix)} chars")
    
    # 计算 Embedding 相似度
    print(f"\n[6] Computing Embedding Similarity...")
    sim_matrix, embeddings = compute_embedding_similarity(prefix_texts, embed_model)
    
    # 绘制热力图
    print(f"\n[7] Generating visualizations")
    plot_heatmap(
        sim_matrix,
        node_names,
        f'Embedding Prefix Similarity (Cosine)\nQuestion: "{question_text[:50]}..."',
        os.path.join(OUTPUT_DIR, 'embedding_similarity.png')
    )
    
    # 绘制 2D 降维图
    plot_2d_embedding(
        embeddings,
        labels,
        node_names,
        f'Prefix Embedding 2D (PCA)\nQuestion: "{question_text[:50]}..."',
        os.path.join(OUTPUT_DIR, 'embedding_2d_pca.png')
    )
    
    # 统计
    upper_tri = np.triu_indices_from(sim_matrix, k=1)
    print(f"\n[8] Statistics:")
    print(f"    Mean: {np.mean(sim_matrix[upper_tri]):.4f}")
    print(f"    Max:  {np.max(sim_matrix[upper_tri]):.4f}")
    print(f"    Min:  {np.min(sim_matrix[upper_tri]):.4f}")
    
    print(f"\n[Done] Figures saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
