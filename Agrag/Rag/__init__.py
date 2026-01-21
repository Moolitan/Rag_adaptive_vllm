"""
Agrag/Rag - 独立版本的RAG实现

每个RAG实现都是独立的Python文件，不依赖工厂模式，
各自包含自己的状态定义、提示词、链、节点、边等组件。

可用的RAG实现：
- ag_rag: 基础RAG（向量检索 → 生成）
- c_rag: Corrective RAG（带评分、web搜索补充、质量检查）
- hop2_rag: 多跳RAG（性能优化版，使用reranker和sentence selector）
- self_rag: 自我反思RAG（带文档和生成反思机制）
"""

from .ag_rag import run_ag_rag, get_ag_rag_app, build_ag_rag
from .c_rag import run_c_rag, get_c_rag_app, build_c_rag
from .hop2_rag import (
    run_hop2_rag,
    get_hop2_rag_app,
    build_hop2_rag,
    enable_instrumentation,
    disable_instrumentation,
    clear_instrumentation_log,
)
from .self_rag import run_self_rag, get_self_rag_app, build_self_rag

__all__ = [
    # AgRag
    "run_ag_rag",
    "get_ag_rag_app",
    "build_ag_rag",
    # CRag
    "run_c_rag",
    "get_c_rag_app",
    "build_c_rag",
    # Hop2Rag
    "run_hop2_rag",
    "get_hop2_rag_app",
    "build_hop2_rag",
    "enable_instrumentation",
    "disable_instrumentation",
    "clear_instrumentation_log",
    # SelfRag
    "run_self_rag",
    "get_self_rag_app",
    "build_self_rag",
]
