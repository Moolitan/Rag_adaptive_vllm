"""
CRag - Corrective RAG实现（并发版本）
流程：LLM判断数据源 → 检索 → 文档评分过滤 → 生成 → 质量检查 → 重试机制

支持传入独立的 DataCollector，用于并发测试场景
使用线程安全的单例模式管理资源
"""
from typing import TypedDict, List, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.graph import StateGraph, START, END
from threading import Lock

from runner.LanggraphMonitor import DataCollector


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
ROUTER_PROMPT = """You are an expert router deciding whether to answer a question using a local Wikipedia-based vectorstore or web search.

The local vectorstore contains a large snapshot of Wikipedia, covering:
- General world knowledge
- Historical events
- People, places, organizations
- Science, technology, politics, culture, arts, sports
- Well-established facts and concepts
- Non-time-sensitive information

Use the vectorstore for:
- Questions about well-known people, places, events, or concepts
- Historical facts or background information
- Definitions, explanations, or descriptions
- Scientific or technical concepts that are not cutting-edge
- Questions that could reasonably be answered by Wikipedia
- Ambiguous or underspecified questions that do NOT clearly require up-to-date information

Use web search ONLY for:
- Current or breaking news
- Events or facts after the Wikipedia snapshot was created
- Rapidly changing information (e.g., stock prices, elections in progress, live scores)
- Newly released papers, products, or technologies
- Information explicitly requiring "latest", "recent", "today", or real-time data

If unsure, prefer the vectorstore.

Return a JSON object with a single key "datasource" set to either "vectorstore" or "web_search".

Question to route:
{question}
"""


# RAG生成提示词
RAG_PROMPT = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Keep the answer concise and relevant.

=== Retrieved Context ===
{context}

Question: {question}

Answer:"""

# 文档评分提示词（批量版本）
DOC_GRADER_PROMPT = """You are a grader assessing relevance of multiple retrieved documents to a user question.

A document is relevant if it contains ANY information that could help answer the question,
including partial information, background context, or related entities mentioned in the question.

Be LENIENT: If the document mentions any person, place, or concept from the question, mark it as relevant.

Documents to evaluate:
{documents}

Question: {question}

Return a JSON object with a single key "scores" containing an array of "yes" or "no" values,
one for each document in the same order they were provided.
Example: {{"scores": ["yes", "no", "yes", "yes", "no"]}}

Use "yes" if the document is even partially relevant."""

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
# 线程安全的资源管理（并发版本）
# =============================

# 全局单例 Embedding 模型（线程安全）
_embedding_model = None
_embedding_lock = Lock()

# Retriever 缓存（线程安全）
_retriever = None
_retriever_lock = Lock()

# LLM 链缓存（线程安全）
_router_chain = None
_rag_chain = None
_doc_grader = None
_hallucination_grader = None
_answer_grader = None
_chain_lock = Lock()

# Web搜索工具（线程安全）
_web_search_tool = None
_web_search_lock = Lock()


def get_embedding_model():
    """获取全局单例 Embedding 模型（线程安全）"""
    global _embedding_model
    with _embedding_lock:
        if _embedding_model is None:
            from langchain_huggingface import HuggingFaceEmbeddings
            print("[INFO] Loading embedding model (first time)...")
            _embedding_model = HuggingFaceEmbeddings(
                model_name="/mnt/Large_Language_Model_Lab_1/模型/rag_models/BAAI-bge-base-en-v1.5",
                model_kwargs={"device": "cuda"},
                encode_kwargs={"normalize_embeddings": True},
            )
            print("[INFO] Embedding model loaded successfully!")
        return _embedding_model


def get_retriever(debug: bool = False, use_remote: bool = True):
    """
    获取 retriever 实例（线程安全）
    
    Args:
        debug: 是否开启调试模式
        use_remote: 是否优先使用远程服务（默认 True）
    
    Returns:
        retriever 实例（远程或本地）
    """
    global _retriever
    
    with _retriever_lock:
        # 如果已经初始化过，直接返回缓存的实例
        if _retriever is not None:
            return _retriever
        
        # 优先尝试连接远程服务
        if use_remote:
            try:
                from Rag.faiss_client import RemoteRetriever
                remote = RemoteRetriever()
                if remote.is_available():
                    print("[FAISS] Using remote retriever service")
                    _retriever = remote
                    return _retriever
                else:
                    print("[FAISS] Remote service not available, falling back to local")
            except Exception as e:
                print(f"[FAISS] Failed to connect remote: {e}, falling back to local")
        
        # 回退到本地初始化
        import os
        from langchain_community.vectorstores import FAISS

        faiss_dir = os.environ.get("AGRAG_FAISS_DIR")
        if not faiss_dir:
            raise RuntimeError("Please set AGRAG_FAISS_DIR")

        print("[FAISS] Loading local database...")
        # 使用全局单例 embedding 模型
        embedding = get_embedding_model()

        vectorstore = FAISS.load_local(
            faiss_dir,
            embedding,
            allow_dangerous_deserialization=True,
        )
        print("[FAISS] Local database loaded")

        retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

        if debug:
            def _debug_get_docs(query: str):
                docs = retriever.get_relevant_documents(query)
                print(f"[FAISS] query='{query}'")
                print(f"[FAISS] retrieved {len(docs)} docs")
                for i, d in enumerate(docs[:3]):
                    print(f"--- doc {i} ---")
                    print(d.page_content[:200])
                    print(d.metadata)
                return docs
            retriever.get_relevant_documents = _debug_get_docs  # monkey patch

        # 缓存实例
        _retriever = retriever
        return _retriever


def init_retriever(debug: bool = False, use_remote: bool = True):
    """
    显式预初始化 retriever（线程安全）
    
    Args:
        debug: 调试模式
        use_remote: 是否使用远程服务
    
    Returns:
        初始化后的 retriever 实例
    """
    return get_retriever(debug=debug, use_remote=use_remote)


# =============================
# Web搜索工具（线程安全）
# =============================

def get_web_search_tool():
    """获取web搜索工具（线程安全）"""
    global _web_search_tool
    with _web_search_lock:
        if _web_search_tool is None:
            from langchain_tavily import TavilySearch
            _web_search_tool = TavilySearch(max_results=5)
        return _web_search_tool


# =============================
# 链构建（线程安全）
# =============================

def get_router_chain():
    """获取路由链（线程安全）"""
    global _router_chain
    with _chain_lock:
        if _router_chain is None:
            prompt = PromptTemplate(template=ROUTER_PROMPT, input_variables=["question"])
            _router_chain = prompt | get_llm(json_mode=True) | JsonOutputParser()
        return _router_chain


def get_rag_chain():
    """获取RAG生成链（线程安全）"""
    global _rag_chain
    with _chain_lock:
        if _rag_chain is None:
            prompt = PromptTemplate(template=RAG_PROMPT, input_variables=["question", "context"])
            _rag_chain = prompt | get_llm() | StrOutputParser()
        return _rag_chain


def get_doc_grader():
    """获取文档评分器（线程安全，批量版本）"""
    global _doc_grader
    with _chain_lock:
        if _doc_grader is None:
            prompt = PromptTemplate(template=DOC_GRADER_PROMPT, input_variables=["question", "documents"])
            _doc_grader = prompt | get_llm(json_mode=True) | JsonOutputParser()
        return _doc_grader


def get_hallucination_grader():
    """获取幻觉检测器（线程安全）"""
    global _hallucination_grader
    with _chain_lock:
        if _hallucination_grader is None:
            prompt = PromptTemplate(template=HALLUCINATION_PROMPT, input_variables=["generation", "documents"])
            _hallucination_grader = prompt | get_llm(json_mode=True) | JsonOutputParser()
        return _hallucination_grader


def get_answer_grader():
    """获取答案评分器（线程安全）"""
    global _answer_grader
    with _chain_lock:
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
    retriever = get_retriever(debug = False)
    docs = retriever.invoke(state["question"])
    return {
        "documents": docs,
        "data_source": "vectorstore",
        "used_web_search": False
    }


def grade_documents_node(state: CRagState) -> Dict[str, Any]:
    """文档评分节点：批量评估所有文档的相关性（单次LLM调用）"""
    docs = state.get("documents", [])
    
    if not docs:
        return {"document_scores": []}
    
    grader = get_doc_grader()
    
    # 构建带编号的文档列表字符串
    docs_text = "\n\n".join([
        f"[Document {i+1}]:\n{d.page_content[:1000]}"  # 限制每个文档长度避免超出上下文
        for i, d in enumerate(docs)
    ])
    
    # 单次调用LLM评估所有文档
    result = grader.invoke({
        "question": state["question"],
        "documents": docs_text
    })
    
    # 解析批量评分结果
    score_list = result.get("scores", [])
    
    # 如果返回的分数数量不匹配，用 "no" 填充或截断
    while len(score_list) < len(docs):
        score_list.append("no")
    score_list = score_list[:len(docs)]
    
    scores = [
        {"document": d, "score": score_list[i]}
        for i, d in enumerate(docs)
    ]

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
# 资源预热（并发版本）
# =============================

def warmup_resources(use_remote: bool = True):
    """
    预热资源：在并发测试开始前加载所有必要的资源

    Args:
        use_remote: 是否使用远程 FAISS 服务
    """
    print("\n" + "=" * 70)
    print("Warming up resources...")
    print("=" * 70)

    import time
    start_time = time.time()

    # 1. 预加载 Embedding 模型
    print("[1/4] Loading embedding model...")
    embedding_start = time.time()
    _ = get_embedding_model()
    embedding_time = time.time() - embedding_start
    print(f"      Embedding model loaded in {embedding_time:.2f}s")

    # 2. 预加载 Retriever
    print("[2/4] Loading retriever...")
    retriever_start = time.time()
    retriever = get_retriever(debug=False, use_remote=use_remote)
    retriever_time = time.time() - retriever_start
    print(f"      Retriever loaded in {retriever_time:.2f}s")

    # 3. 预加载 LLM 链
    print("[3/4] Loading LLM chains...")
    llm_start = time.time()
    _ = get_router_chain()
    _ = get_rag_chain()
    _ = get_doc_grader()
    _ = get_hallucination_grader()
    _ = get_answer_grader()
    llm_time = time.time() - llm_start
    print(f"      LLM chains loaded in {llm_time:.2f}s")

    # 4. 预热向量数据库（执行实际检索操作）
    print("[4/4] Warming up vector database...")
    db_start = time.time()
    try:
        # 执行一次实际检索，触发向量数据库加载索引到内存
        _ = retriever.invoke("warmup query")
        db_time = time.time() - db_start
        print(f"      Vector database warmed up in {db_time:.2f}s")
    except Exception as e:
        db_time = time.time() - db_start
        print(f"      Vector database warmup failed in {db_time:.2f}s: {e}")

    total_time = time.time() - start_time
    print("=" * 70)
    print(f"Warmup complete! Total time: {total_time:.2f}s")
    print("=" * 70 + "\n")


# =============================
# 便捷调用接口（并发版本）
# =============================

def run_c_rag_with_collector(
    question: str,
    collector: DataCollector = None
) -> Dict[str, Any]:
    """
    运行 CRag（支持传入独立的 DataCollector）

    Args:
        question: 用户问题
        collector: DataCollector 实例（如果为 None，创建临时的）

    Returns:
        包含 answer, metadata, documents 的字典
    """
    app = get_c_rag_app()

    # 如果没有提供 collector，创建临时的
    if collector is None:
        collector = DataCollector(track_prompts=False)

    result = app.invoke(
        {"question": question},
        config={
            "callbacks": [collector],
            "recursion_limit": 100
        }
    )

    return {
        "answer": result.get("final_answer", ""),
        "metadata": result.get("metadata", {}),
        "documents": result.get("documents", [])
    }


def run_c_rag(question: str) -> Dict[str, Any]:
    """
    运行CRag查询（兼容旧接口，使用临时 collector）

    Args:
        question: 用户问题

    Returns:
        包含 answer, metadata, documents 的字典
    """
    return run_c_rag_with_collector(question, collector=None)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
    else:
        q = "What are the latest developments in AI?"

    print(f"Question: {q}")

    # 创建独立的 collector
    collector = DataCollector(track_prompts=True, encoding_name="cl100k_base")

    result = run_c_rag_with_collector(
        question=q,
        collector=collector
    )

    print(f"Answer: {result['answer']}")
    print(f"Metadata: {result['metadata']}")

    # 查看 LLM calls
    llm_calls = collector.get_llm_calls()
    print(f"\nLLM calls: {len(llm_calls)}")
    for i, call in enumerate(llm_calls[:3], 1):
        print(f"  {i}. Node: {call['node_name']}, Tokens: {call['prompt_tokens']}")
