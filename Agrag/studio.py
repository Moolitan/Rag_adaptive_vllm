# -*- coding: utf-8 -*-
"""
================================================================================
studio.py - LangGraph Studio 入口文件

【文件概述】
这是 LangGraph Studio 的入口文件，用于在 LangGraph Studio 中可视化和调试 RAG 工作流。

【使用方式】
1. 确保已安装 langgraph-cli:
   pip install langgraph-cli

2. 在项目根目录运行:
   langgraph dev

3. 浏览器会自动打开 Studio 界面

【支持的工作流】
- hop2rag: 多跳RAG，适用于需要多步推理的复杂问题
- crag: 纠正式RAG，带有文档相关性检查和网络搜索回退
- agrag: 自适应RAG，根据问题类型动态选择处理策略
- selfrag: 自反思RAG，带有生成质量自我评估

作者: [Your Name]
================================================================================
"""

from Rag.hop2_rag2 import get_hop2_rag_app
from Rag.c_rag import get_c_rag_app
from Rag.ag_rag import get_ag_rag_app
from Rag.self_rag import get_self_rag_app

# =============================
# 导出编译后的工作流图
# =============================

# 多跳RAG - 需要多步推理的复杂问题
hop2rag = get_hop2_rag_app()

# 纠正式RAG - 带文档相关性检查
crag = get_c_rag_app()

# 自适应RAG - 根据问题类型动态选择策略
agrag = get_ag_rag_app()

# 自反思RAG - 带生成质量自我评估
selfrag = get_self_rag_app()


if __name__ == "__main__":
    """快速测试工作流是否正常加载"""
    print("=" * 50)
    print("LangGraph Studio 工作流加载测试")
    print("=" * 50)

    workflows = {
        "hop2rag": hop2rag,
        "crag": crag,
        "agrag": agrag,
        "selfrag": selfrag
    }

    for name, workflow in workflows.items():
        try:
            nodes = list(workflow.get_graph().nodes.keys())
            print(f"\n[{name}] 加载成功")
            print(f"  节点数量: {len(nodes)}")
            print(f"  节点列表: {nodes}")
        except Exception as e:
            print(f"\n[{name}] 加载失败: {e}")

    print("\n" + "=" * 50)
    print("测试完成! 可以运行 'langgraph dev' 启动 Studio")
    print("=" * 50)
