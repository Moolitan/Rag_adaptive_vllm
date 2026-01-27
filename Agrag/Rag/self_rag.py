from typing import TypedDict, List, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.graph import StateGraph, START, END

# =============================
# 状态定义
# =============================

class SelfRagState(TypedDict, total=False):
    """SelfRag状态定义"""
    # 核心输入输出
    question: str                         # 用户问题
    generation: str                       # 生成的答案
    documents: List[Document]             # 当前文档列表

    # 数据源信息
    data_source: str                       # 使用的数据源
    used_web_search: bool                  # 是否使用了web搜索

    # 自我反思字段（文档）
    reflection_docs_relevant_count: int    # 相关文档数量（自我反思）
    reflection_need_more_info: bool        # 是否需要更多信息

    # 自我反思字段（生成）
    grounded: str                          # "yes" 或 "no" - 答案是否有依据
    useful: str                            # "yes" 或 "no" - 答案是否有用
    reflection_need_improvement: bool      # 生成是否需要改进

    # 循环控制
    loop: int                              # 循环计数器
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

# 自我反思RAG生成提示词
SELF_RAG_PROMPT = """You are an assistant for question-answering tasks with self-reflection capability.
Use the following pieces of retrieved context to answer the question.
Think carefully about whether the context provides sufficient information.
If you don't know the answer or the context is insufficient, acknowledge it honestly.

=== Retrieved Context ===
{context}

Question: {question}

Instructions:
1. First, assess if the retrieved documents are relevant and sufficient
2. If relevant, synthesize the information to answer the question
3. If insufficient, indicate what additional information would be needed
4. Be concise but thorough in your answer

Answer:"""

# 文档相关性评分提示词
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

Consider:
1. Does the answer directly address the question?
2. Is the answer informative and helpful?
3. Does it provide actionable or meaningful information?

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
    """延迟加载retriever"""
    global _retriever
    if _retriever is None:
        import os
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma

        persist_dir = os.environ.get("AGRAG_PERSIST_DIR")
        collection_name = os.environ.get("AGRAG_COLLECTION_NAME")

        if not persist_dir or not collection_name:
            raise RuntimeError("请设置环境变量 AGRAG_PERSIST_DIR 和 AGRAG_COLLECTION_NAME")

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

_self_rag_chain = None
_doc_grader = None
_hallucination_grader = None
_answer_grader = None


def get_self_rag_chain():
    """获取自我反思RAG生成链"""
    global _self_rag_chain
    if _self_rag_chain is None:
        prompt = PromptTemplate(
            template=SELF_RAG_PROMPT,
            input_variables=["question", "context"]
        )
        _self_rag_chain = prompt | get_llm() | StrOutputParser()
    return _self_rag_chain


def get_doc_grader():
    """获取文档评分器"""
    global _doc_grader
    if _doc_grader is None:
        prompt = PromptTemplate(
            template=DOC_GRADER_PROMPT,
            input_variables=["question", "document"]
        )
        _doc_grader = prompt | get_llm(json_mode=True) | JsonOutputParser()
    return _doc_grader


def get_hallucination_grader():
    """获取幻觉检测器"""
    global _hallucination_grader
    if _hallucination_grader is None:
        prompt = PromptTemplate(
            template=HALLUCINATION_PROMPT,
            input_variables=["generation", "documents"]
        )
        _hallucination_grader = prompt | get_llm(json_mode=True) | JsonOutputParser()
    return _hallucination_grader


def get_answer_grader():
    """获取答案评分器"""
    global _answer_grader
    if _answer_grader is None:
        prompt = PromptTemplate(
            template=ANSWER_GRADER_PROMPT,
            input_variables=["generation", "question"]
        )
        _answer_grader = prompt | get_llm(json_mode=True) | JsonOutputParser()
    return _answer_grader


# =============================
# 节点函数
# =============================

def retrieve_node(state: SelfRagState) -> Dict[str, Any]:
    """检索节点：从向量库检索"""
    retriever = get_retriever()
    docs = retriever.invoke(state["question"])
    return {
        "documents": docs,
        "data_source": "vectorstore",
        "used_web_search": False
    }


def self_reflect_docs_node(state: SelfRagState) -> Dict[str, Any]:
    """自我反思文档节点：评估检索文档质量"""
    docs = state.get("documents", [])
    grader = get_doc_grader()

    # 评估每个文档的相关性
    relevant_count = 0
    for d in docs:
        score = grader.invoke({
            "question": state["question"],
            "document": d.page_content
        })
        if score.get("score") == "yes":
            relevant_count += 1

    # 判断是否需要更多信息（少于2个相关文档）
    need_more_info = relevant_count < 2

    return {
        "reflection_docs_relevant_count": relevant_count,
        "reflection_need_more_info": need_more_info
    }


def web_search_node(state: SelfRagState) -> Dict[str, Any]:
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


def generate_node(state: SelfRagState) -> Dict[str, Any]:
    """生成节点：带自我反思的生成"""
    documents = state.get("documents", [])
    context = "\n\n".join([d.page_content for d in documents])

    rag_chain = get_self_rag_chain()
    result = rag_chain.invoke({
        "question": state["question"],
        "context": context
    })

    return {"generation": result}


def self_reflect_generation_node(state: SelfRagState) -> Dict[str, Any]:
    """自我反思生成节点：评估生成答案质量"""
    hallucination_grader = get_hallucination_grader()
    answer_grader = get_answer_grader()

    # 评估答案是否有依据
    grounded_score = hallucination_grader.invoke({
        "generation": state["generation"],
        "documents": state["documents"]
    })

    # 评估答案是否有用
    useful_score = answer_grader.invoke({
        "generation": state["generation"],
        "question": state["question"]
    })

    grounded = grounded_score.get("score", "no")
    useful = useful_score.get("score", "no")

    # 判断是否需要改进
    need_improvement = grounded != "yes" or useful != "yes"
    loop = int(state.get("loop", 0) or 0)
    # 仅在需要改进时递增循环计数
    next_loop = loop + 1 if need_improvement else loop

    return {
        "grounded": grounded,
        "useful": useful,
        "reflection_need_improvement": need_improvement,
        "loop": next_loop,
    }


def finalize_node(state: SelfRagState) -> Dict[str, Any]:
    """最终化节点：整理输出"""
    return {
        "final_answer": state.get("generation", ""),
        "metadata": {
            "data_source": state.get("data_source"),
            "used_web_search": state.get("used_web_search", False),
            "web_search_rounds": state.get("web_search_rounds", 0),
            "grounded": state.get("grounded"),
            "useful": state.get("useful"),
            "reflection_docs_relevant_count": state.get("reflection_docs_relevant_count", 0),
            "loop_count": state.get("loop", 0),
        }
    }


# =============================
# 边函数（条件路由）
# =============================

def route_by_reflection_docs(state: SelfRagState) -> str:
    """根据文档自我反思结果选择路径"""
    need_more_info = state.get("reflection_need_more_info", False)

    if need_more_info:
        # 文档不足，需要web搜索
        return "web_search"
    else:
        # 文档充足，进行生成
        return "generate"


def route_by_reflection_generation(state: SelfRagState) -> str:
    """根据生成自我反思结果选择路径"""
    need_improvement = state.get("reflection_need_improvement", False)
    loop_count = state.get("loop", 0)

    # 重试限制（最多2次循环）
    max_loops = 2
    can_retry = loop_count < max_loops

    if need_improvement and can_retry:
        # 生成需要改进且可以重试
        return "retrieve"
    else:
        # 质量满足或已用尽重试次数
        return "finalize"


# =============================
# 构建图
# =============================

def build_self_rag() -> StateGraph:
    """构建SelfRag工作流"""
    workflow = StateGraph(SelfRagState)

    # 添加节点
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("self_reflect_docs", self_reflect_docs_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("self_reflect_generation", self_reflect_generation_node)
    workflow.add_node("finalize", finalize_node)

    # 入口
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "self_reflect_docs")

    # 条件边1：自我反思文档
    workflow.add_conditional_edges(
        "self_reflect_docs",
        route_by_reflection_docs,
        {"web_search": "web_search", "generate": "generate"}
    )

    # web搜索后进入生成
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", "self_reflect_generation")

    # 条件边2：自我反思生成
    workflow.add_conditional_edges(
        "self_reflect_generation",
        route_by_reflection_generation,
        {"finalize": "finalize", "retrieve": "retrieve"}
    )

    # 出口
    workflow.add_edge("finalize", END)

    return workflow


def get_self_rag_app():
    """获取编译后的SelfRag应用"""
    workflow = build_self_rag()
    return workflow.compile()


# =============================
# 便捷调用接口
# =============================

def run_self_rag(question: str) -> Dict[str, Any]:
    """
    运行SelfRag查询

    Args:
        question: 用户问题

    Returns:
        包含 final_answer 和 metadata 的字典
    """
    app = get_self_rag_app()
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
        q = "What is self-reflective RAG and how does it work?"

    print(f"Question: {q}")
    result = run_self_rag(q)
    print(f"Answer: {result['answer']}")
    print(f"Metadata: {result['metadata']}")
