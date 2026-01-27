"""
CRag - Corrective RAG实现（独立版本）
流程：LLM判断数据源 → 检索 → 文档评分过滤 → 生成 → 质量检查 → 重试机制
不依赖工厂模式，所有组件独立实现
"""
from typing import TypedDict, List, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.graph import StateGraph, START, END

from runner.LanggraphMonitor import DataCollector

# 全局性能监控器实例
monitor = DataCollector()

def get_performance_records():
    """获取性能监控记录"""
    return monitor.records

def clear_performance_records():
    """清空性能监控记录"""
    monitor.records.clear()
    monitor.starts.clear()


# =============================
# 状态定义
# =============================

class CRagState(TypedDict, total=False):
    """CRag状态定义"""
    # 核心输入输出
    question: str                         # 用户问题
    generation: str                       # 生成的答案
    documents: List[Document]             # 当前文档列表

    # 路由决策
    router_decision: str                   # "web_search" 或 "vectorstore"
    data_source: str                       # 实际使用的数据源
    used_web_search: bool                  # 是否使用了web搜索

    # 文档评分
    document_scores: List[Dict[str, Any]]  # 文档评分列表
    graded_relevant_docs: int              # 相关文档数量
    has_irrelevant_docs: bool              # 是否有不相关文档

    # 生成评分
    grounded: str                          # "yes" 或 "no" - 答案是否有依据
    useful: str                            # "yes" 或 "no" - 答案是否有用

    # 循环控制
    web_search_rounds: int                 # web搜索轮次

    # 额外数据
    web_docs: List[Document]               # web搜索结果

    # 最终输出
    final_answer: str                      # 最终答案
    metadata: Dict[str, Any]               # 元数据


# =============================
# 配置和LLM
# =============================

from langchain_openai import ChatOpenAI

VLLM_MODEL_NAME = "Qwen2.5"
VLLM_API_BASE = "http://localhost:8000/v1"
VLLM_API_KEY = "EMPTY"


def get_llm(json_mode: bool = False, temperature: float = 0):
    """获取LLM实例"""
    model_kwargs = {}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}
    return ChatOpenAI(
        model=VLLM_MODEL_NAME,
        openai_api_key=VLLM_API_KEY,
        openai_api_base=VLLM_API_BASE,
        temperature=temperature,
        model_kwargs=model_kwargs,
    )


# =============================
# 提示词定义
# =============================

# 路由提示词
ROUTER_PROMPT = """You are an expert at routing a user question to a vectorstore, web search, or general generation.
Use the vectorstore for questions about: LLM agents, prompt engineering, adversarial attacks,
LangGraph/LangChain RAG patterns, and anything likely covered by the local knowledge base.
Use web-search for: current events, newly released papers, unknown entities, time-sensitive facts,
or anything not likely in the local knowledge base.
Return a JSON object with a single key 'datasource' equal to 'web_search' or 'vectorstore'.
Question to route: {question}"""

# RAG生成提示词
RAG_PROMPT = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Keep the answer concise and relevant.

=== Retrieved Context ===
{context}

Question: {question}

Answer:"""

# 文档评分提示词
DOC_GRADER_PROMPT = """You are a grader assessing relevance of a retrieved document to a user question.

A document is relevant if it contains ANY information that could help answer the question,
including partial information, background context, or related entities mentioned in the question.

Be LENIENT: If the document mentions any person, place, or concept from the question, mark it as relevant.

Document:
{document}

Question: {question}

Return JSON {{"score": "yes" or "no"}}. Use "yes" if the document is even partially relevant."""

# 幻觉检测提示词
HALLUCINATION_PROMPT = """You are a grader assessing whether an answer is grounded in / supported by a set of facts.

An answer is GROUNDED if:
1. All claims in the answer can be verified from the facts (direct or inferred)
2. The answer does not contradict the facts
3. Logical inferences based on the facts are acceptable

An answer is NOT GROUNDED if:
1. It makes claims not supported by the facts
2. It contradicts the facts
3. It invents information not present in the facts

Facts:
{documents}

Answer:
{generation}

Return JSON {{"score": "yes" or "no"}}. Use "yes" if the answer is grounded."""

# 答案有用性提示词
ANSWER_GRADER_PROMPT = """You are a grader assessing whether an answer is useful to resolve a question.
Question:
{question}

Answer:
{generation}

Return JSON {{"score": "yes" or "no"}}."""


# =============================
# 向量库配置（延迟导入）
# =============================

_retriever = None


def get_retriever():
    """延迟加载 FAISS retriever"""
    global _retriever
    if _retriever is None:
        import os
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS

        faiss_dir = os.environ.get("AGRAG_FAISS_DIR")

        if not faiss_dir:
            raise RuntimeError("请设置环境变量 AGRAG_FAISS_DIR")

        # 与构建 FAISS 时完全一致的 embedding
        embedding = HuggingFaceEmbeddings(
            model_name="/mnt/Large_Language_Model_Lab_1/模型/rag_models/BAAI-bge-base-en-v1.5",
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True},
        )

        # 加载已存在的 FAISS 向量库
        vectorstore = FAISS.load_local(
            faiss_dir,
            embedding,
            allow_dangerous_deserialization=True
        )

        _retriever = vectorstore.as_retriever()

    return _retriever



# =============================
# Web搜索工具
# =============================

_web_search_tool = None


def get_web_search_tool():
    """获取web搜索工具"""
    global _web_search_tool
    if _web_search_tool is None:
        from langchain_tavily import TavilySearch
        _web_search_tool = TavilySearch(max_results=5)
    return _web_search_tool


# =============================
# 链构建
# =============================

_router_chain = None
_rag_chain = None
_doc_grader = None
_hallucination_grader = None
_answer_grader = None


def get_router_chain():
    """获取路由链"""
    global _router_chain
    if _router_chain is None:
        prompt = PromptTemplate(template=ROUTER_PROMPT, input_variables=["question"])
        _router_chain = prompt | get_llm(json_mode=True) | JsonOutputParser()
    return _router_chain


def get_rag_chain():
    """获取RAG生成链"""
    global _rag_chain
    if _rag_chain is None:
        prompt = PromptTemplate(template=RAG_PROMPT, input_variables=["question", "context"])
        _rag_chain = prompt | get_llm() | StrOutputParser()
    return _rag_chain


def get_doc_grader():
    """获取文档评分器"""
    global _doc_grader
    if _doc_grader is None:
        prompt = PromptTemplate(template=DOC_GRADER_PROMPT, input_variables=["question", "document"])
        _doc_grader = prompt | get_llm(json_mode=True) | JsonOutputParser()
    return _doc_grader


def get_hallucination_grader():
    """获取幻觉检测器"""
    global _hallucination_grader
    if _hallucination_grader is None:
        prompt = PromptTemplate(template=HALLUCINATION_PROMPT, input_variables=["generation", "documents"])
        _hallucination_grader = prompt | get_llm(json_mode=True) | JsonOutputParser()
    return _hallucination_grader


def get_answer_grader():
    """获取答案评分器"""
    global _answer_grader
    if _answer_grader is None:
        prompt = PromptTemplate(template=ANSWER_GRADER_PROMPT, input_variables=["generation", "question"])
        _answer_grader = prompt | get_llm(json_mode=True) | JsonOutputParser()
    return _answer_grader


# =============================
# 节点函数
# =============================

def analyze_question_node(state: CRagState) -> Dict[str, Any]:
    """分析问题节点：LLM判断数据源"""
    router = get_router_chain()
    decision = router.invoke({"question": state["question"]})
    return {"router_decision": decision.get("datasource", "vectorstore")}


def retrieve_node(state: CRagState) -> Dict[str, Any]:
    """检索节点：从向量库检索"""
    retriever = get_retriever()
    docs = retriever.invoke(state["question"])
    return {
        "documents": docs,
        "data_source": "vectorstore",
        "used_web_search": False
    }


def grade_documents_node(state: CRagState) -> Dict[str, Any]:
    """文档评分节点：评估每个文档的相关性"""
    docs = state.get("documents", [])
    grader = get_doc_grader()

    scores = []
    for d in docs:
        score = grader.invoke({
            "question": state["question"],
            "document": d.page_content
        })
        scores.append({
            "document": d,
            "score": score.get("score", "no")
        })

    return {"document_scores": scores}


def filter_documents_node(state: CRagState) -> Dict[str, Any]:
    """过滤文档节点：筛选相关文档"""
    scores = state.get("document_scores", [])

    kept = [s["document"] for s in scores if s["score"] == "yes"]
    has_irrelevant = len(kept) < len(scores)

    return {
        "documents": kept,
        "graded_relevant_docs": len(kept),
        "has_irrelevant_docs": has_irrelevant
    }


def web_search_node(state: CRagState) -> Dict[str, Any]:
    """Web搜索节点"""
    rounds = int(state.get("web_search_rounds", 0) or 0) + 1

    web_tool = get_web_search_tool()
    results = web_tool.invoke(state["question"])

    # 解析结果为Document列表
    docs = []
    if isinstance(results, dict):
        docs = [
            Document(
                page_content=r.get("content", ""),
                metadata={"url": r.get("url", "")}
            )
            for r in results.get("results", [])
        ]
    elif isinstance(results, list):
        docs = [
            Document(
                page_content=r.get("content", "") if isinstance(r, dict) else str(r),
                metadata={"url": r.get("url", "")} if isinstance(r, dict) else {}
            )
            for r in results
        ]
    else:
        docs = [Document(page_content=str(results))]

    return {
        "documents": docs,
        "web_docs": docs,
        "data_source": "web_search",
        "used_web_search": True,
        "web_search_rounds": rounds
    }


def generate_node(state: CRagState) -> Dict[str, Any]:
    """生成节点：基于文档生成答案"""
    documents = state.get("documents", [])
    context = "\n\n".join([d.page_content for d in documents])

    rag_chain = get_rag_chain()
    result = rag_chain.invoke({
        "question": state["question"],
        "context": context
    })

    return {"generation": result}


def grade_generation_node(state: CRagState) -> Dict[str, Any]:
    """生成评分节点：双维度评估"""
    hallucination_grader = get_hallucination_grader()
    answer_grader = get_answer_grader()

    grounded_score = hallucination_grader.invoke({
        "generation": state["generation"],
        "documents": state["documents"]
    })

    useful_score = answer_grader.invoke({
        "generation": state["generation"],
        "question": state["question"]
    })

    return {
        "grounded": grounded_score.get("score", "no"),
        "useful": useful_score.get("score", "no")
    }


def finalize_node(state: CRagState) -> Dict[str, Any]:
    """最终化节点：整理输出"""
    return {
        "final_answer": state.get("generation", ""),
        "metadata": {
            "data_source": state.get("data_source"),
            "used_web_search": state.get("used_web_search", False),
            "web_search_rounds": state.get("web_search_rounds", 0),
            "grounded": state.get("grounded"),
            "useful": state.get("useful"),
            "graded_relevant_docs": state.get("graded_relevant_docs", 0),
        }
    }


# =============================
# 边函数（条件路由）
# =============================

def route_by_decision(state: CRagState) -> str:
    """根据LLM路由决策选择路径"""
    decision = state.get("router_decision", "vectorstore")
    if decision == "web_search":
        return "web_search"
    return "vectorstore"


def route_by_relevance(state: CRagState) -> str:
    """根据文档相关性选择路径"""
    relevant_count = state.get("graded_relevant_docs", 0)
    if relevant_count == 0:
        return "web_search"
    return "generate"


def route_with_retry(state: CRagState) -> str:
    """根据生成质量和重试次数选择路径"""
    grounded = state.get("grounded", "no")
    useful = state.get("useful", "no")
    rounds = state.get("web_search_rounds", 0)

    is_good_quality = (grounded == "yes" and useful == "yes")
    max_retries = 3
    can_retry = rounds < max_retries

    if not is_good_quality and can_retry:
        return "web_search"
    return "finalize"


# =============================
# 构建图
# =============================

def build_c_rag() -> StateGraph:
    """构建CRag工作流"""
    workflow = StateGraph(CRagState)

    # 添加节点
    workflow.add_node("analyze_question", analyze_question_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_docs", grade_documents_node)
    workflow.add_node("filter_docs", filter_documents_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("grade_generation", grade_generation_node)
    workflow.add_node("finalize", finalize_node)

    # 入口：LLM判断数据源
    workflow.add_edge(START, "analyze_question")

    # 条件边1：路由到vectorstore或web_search
    workflow.add_conditional_edges(
        "analyze_question",
        route_by_decision,
        {"web_search": "web_search", "vectorstore": "retrieve"}
    )

    # 固定边：retrieve → grade_docs → filter_docs
    workflow.add_edge("retrieve", "grade_docs")
    workflow.add_edge("grade_docs", "filter_docs")

    # 条件边2：文档评分后路由
    workflow.add_conditional_edges(
        "filter_docs",
        route_by_relevance,
        {"web_search": "web_search", "generate": "generate"}
    )

    # 固定边：web_search → generate
    workflow.add_edge("web_search", "generate")

    # 固定边：generate → grade_generation
    workflow.add_edge("generate", "grade_generation")

    # 条件边3：生成质量检查 + 重试
    workflow.add_conditional_edges(
        "grade_generation",
        route_with_retry,
        {"finalize": "finalize", "web_search": "web_search"}
    )

    # 出口
    workflow.add_edge("finalize", END)

    return workflow


def get_c_rag_app():
    """获取编译后的CRag应用"""
    workflow = build_c_rag()
    return workflow.compile()


# =============================
# 便捷调用接口
# =============================

def run_c_rag(question: str) -> Dict[str, Any]:
    """
    运行CRag查询

    Args:
        question: 用户问题

    Returns:
        包含 final_answer 和 metadata 的字典
    """
    app = get_c_rag_app()
    result = app.invoke({"question": question})
    return {
        "answer": result.get("final_answer", ""),
        "metadata": result.get("metadata", {}),
        "documents": result.get("documents", [])
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
    else:
        q = "What are the latest developments in AI?"

    print(f"Question: {q}")
    result = run_c_rag(q)
    print(f"Answer: {result['answer']}")
    print(f"Metadata: {result['metadata']}")
