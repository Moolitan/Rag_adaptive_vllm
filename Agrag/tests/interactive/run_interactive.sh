#!/bin/bash
# 交互式 RAG 测试脚本
# 环境变量已通过 .env 文件自动加载，无需手动 export

cd "$(dirname "$0")/.."

# 激活 conda 环境
# conda activate langgraph_vllm

# 默认使用 crag
python tests/run_interactive.py --rag crag

# ========================================
# 其他可用命令参考：
# ========================================

# 使用不同的 RAG 变体
# python tests/run_interactive.py --rag agrag
# python tests/run_interactive.py --rag hop2rag
# python tests/run_interactive.py --rag selfrag

# 保存工作流图
# python tests/run_interactive.py --rag hop2rag --save-graph
