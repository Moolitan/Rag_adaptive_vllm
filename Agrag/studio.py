# -*- coding: utf-8 -*-
"""
LangGraph Studio 入口文件
用于在 LangGraph Studio 中可视化和调试 RAG 工作流
"""
from graph.rag import get_rag

# 导出各个 RAG 变体的编译图供 LangGraph Studio 使用
hop2rag = get_rag("hop2rag").build().compile()
crag = get_rag("crag").build().compile()
agrag = get_rag("agrag").build().compile()
selfrag = get_rag("selfrag").build().compile()
