"""
AgRag - 基础RAG实现（独立版本）
简单的线性流程：向量检索 → 生成
不依赖工厂模式，所有组件独立实现
"""
from typing import TypedDict, List, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

# =============================
# 状态定义
# =============================

class AgRagState(TypedDict, total=False):
    """AgRag状态定义"""
    # 核心输入输出
    question: str                         # 用户问题
    generation: str                       # 生成的答案
    documents: List[Document]             # 检索到的文档

    # 数据源信息
    data_source: str                       # 使用的数据源
    used_web_search: bool                  # 是否使用了web搜索

    # 最终输出
    final_answer: str                      # 最终处理后的答案
    metadata: Dict[str, Any]               # 额外元数据


# =============================
# 配置和LLM
# =============================

from langchain_openai import ChatOpenAI

VLLM_MODEL_NAME = "Qwen2.5"
VLLM_API_BASE = "http://localhost:8000/v1"
VLLM_API_KEY = "EMPTY"


def get_llm(temperature: float = 0):
    """获取LLM实例"""
    return ChatOpenAI(
        model=VLLM_MODEL_NAME,
        openai_api_key=VLLM_API_KEY,
        openai_api_base=VLLM_API_BASE,
        temperature=temperature,
    )


# =============================
# 提示词定义
# =============================

AG_RAG_PROMPT = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Keep the answer concise and relevant.

=== Retrieved Context ===
{context}

Question: {question}

Answer:"""


# =============================
# 向量库配置（延迟导入）
# =============================

_retriever = None


def get_retriever():
    """延迟加载retriever"""
    global _retriever
    if _retriever is None:
        import os
        from functools import lru_cache
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma

        persist_dir = os.environ.get("AGRAG_PERSIST_DIR")
        collection_name = os.environ.get("AGRAG_COLLECTION_NAME")

        if not persist_dir or not collection_name:
            raise RuntimeError("请设置环境变量 AGRAG_PERSIST_DIR 和 AGRAG_COLLECTION_NAME")

        # 使用BGE嵌入模型
        embedding = HuggingFaceEmbeddings(
            model_name="/mnt/Large_Language_Model_Lab_1/模型/rag_models/BAAI-bge-base-en-v1.5",
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True},
        )

        vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=persist_dir,
            embedding_function=embedding,
        )
        _retriever = vectorstore.as_retriever()

    return _retriever


# =============================
# RAG链
# =============================

_rag_chain = None


def get_rag_chain():
    """获取RAG生成链"""
    global _rag_chain
    if _rag_chain is None:
        prompt = PromptTemplate(
            template=AG_RAG_PROMPT,
            input_variables=["question", "context"],
        )
        llm = get_llm(temperature=0)
        _rag_chain = prompt | llm | StrOutputParser()
    return _rag_chain


# =============================
# 节点函数
# =============================

def retrieve_node(state: AgRagState) -> Dict[str, Any]:
    """检索节点：从向量库检索相关文档"""
    retriever = get_retriever()
    docs = retriever.invoke(state["question"])

    return {
        "documents": docs,
        "data_source": "vectorstore",
        "used_web_search": False
    }


def generate_node(state: AgRagState) -> Dict[str, Any]:
    """生成节点：基于检索文档生成答案"""
    documents = state.get("documents", [])
    context = "\n\n".join([d.page_content for d in documents])

    rag_chain = get_rag_chain()
    result = rag_chain.invoke({
        "question": state["question"],
        "context": context
    })

    return {"generation": result}


def finalize_node(state: AgRagState) -> Dict[str, Any]:
    """最终化节点：整理输出"""
    return {
        "final_answer": state.get("generation", ""),
        "metadata": {
            "data_source": state.get("data_source"),
            "used_web_search": state.get("used_web_search", False),
        }
    }


# =============================
# 构建图
# =============================

def build_ag_rag() -> StateGraph:
    """构建AgRag工作流"""
    workflow = StateGraph(AgRagState)

    # 添加节点
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("finalize", finalize_node)

    # 简单的线性流程
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "finalize")
    workflow.add_edge("finalize", END)

    return workflow


def get_ag_rag_app():
    """获取编译后的AgRag应用"""
    workflow = build_ag_rag()
    return workflow.compile()


# =============================
# 便捷调用接口
# =============================

def run_ag_rag(question: str) -> Dict[str, Any]:
    """
    运行AgRag查询

    Args:
        question: 用户问题

    Returns:
        包含 final_answer 和 metadata 的字典
    """
    app = get_ag_rag_app()
    result = app.invoke({"question": question})
    return {
        "answer": result.get("final_answer", ""),
        "metadata": result.get("metadata", {}),
        "documents": result.get("documents", [])
    }


if __name__ == "__main__":
    # 简单测试
    import sys
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
    else:
        q = "What is RAG?"

    print(f"Question: {q}")
    result = run_ag_rag(q)
    print(f"Answer: {result['answer']}")
    print(f"Documents retrieved: {len(result['documents'])}")
