"""
CRag2 - Corrective RAG (基于网上成熟代码改造)
保留所有原始逻辑和提示词，仅修改向量库和LLM调用
"""
from typing import Dict, TypedDict, Any, List
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langchain_core.prompts import PromptTemplate
import os
import json
import re

from runner.LanggraphMonitor import DataCollector

# 全局性能监控器实例
monitor = DataCollector()

def get_performance_records():
    """获取性能监控记录"""
    return monitor.get_records()

def clear_performance_records():
    """清空性能监控记录"""
    monitor.clear()


# =============================
# 配置
# =============================

# vLLM配置
VLLM_MODEL_NAME = "Qwen2.5"
VLLM_API_BASE = "http://localhost:8000/v1"
VLLM_API_KEY = "EMPTY"

# FAISS配置
FAISS_DIR = os.environ.get("AGRAG_FAISS_DIR", "")
EMBEDDING_MODEL_PATH = "/mnt/Large_Language_Model_Lab_1/模型/rag_models/BAAI-bge-base-en-v1.5"


# =============================
# Retriever 初始化
# =============================

_retriever = None

def get_retriever(use_remote: bool = True, k: int = 15):
    """
    获取 retriever 实例

    Args:
        use_remote: 是否优先使用远程服务（默认 True）
        k: 检索数量（默认 15）

    Returns:
        retriever 实例（远程或本地）
    """
    global _retriever

    # 优先尝试连接远程服务
    if use_remote:
        try:
            from Rag.faiss_client import RemoteRetriever
            remote = RemoteRetriever()
            if remote.is_available():
                print(f"[FAISS] Using remote retriever service (k={k})")
                remote.search_kwargs = {"k": k}
                return remote
            else:
                print("[FAISS] Remote service not available, falling back to local")
        except Exception as e:
            print(f"[FAISS] Failed to connect remote: {e}, falling back to local")
    
    # 回退到本地初始化
    import os
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings

    faiss_dir = os.environ.get("AGRAG_FAISS_DIR")
    if not faiss_dir:
        raise RuntimeError("Please set AGRAG_FAISS_DIR")

    print("[FAISS] Loading local database...")
    embedding = HuggingFaceEmbeddings(
        model_name="/mnt/Large_Language_Model_Lab_1/模型/rag_models/BAAI-bge-base-en-v1.5",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.load_local(
        faiss_dir,
        embedding,
        allow_dangerous_deserialization=True,
    )
    print("[FAISS] Local database loaded")

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    print(f"[FAISS] Local retriever initialized (k={k})")
    return retriever

def init_retriever(use_remote: bool = True, k: int = 15):
    """初始化 retriever"""
    return get_retriever(use_remote=use_remote, k=k)


# =============================
# Web搜索工具缓存
# =============================

_web_search_tool = None

def get_web_search_tool():
    """获取缓存的 web 搜索工具"""
    global _web_search_tool
    if _web_search_tool is not None:
        return _web_search_tool

    tavily_api_key = os.environ.get("TAVILY_API_KEY", "")
    if not tavily_api_key:
        return None

    # Initialize Tavily search tool (using new API)
    try:
        from langchain_tavily import TavilySearch
        _web_search_tool = TavilySearch(api_key=tavily_api_key, max_results=3)
        print("[Tavily] Using new langchain-tavily API")
    except ImportError:
        # Fallback to old API if new package not installed
        from langchain_community.tools import TavilySearchResults
        _web_search_tool = TavilySearchResults(
            api_key=tavily_api_key,
            max_results=3,
            search_depth="advanced"
        )
        print("[Tavily] Using legacy langchain-community API")

    return _web_search_tool


# =============================
# 状态定义（重构为独立字段）
# =============================

class GraphState(TypedDict, total=False):
    """状态定义 - 将原 keys 字典拆分为独立字段"""
    # 核心字段
    question: str                    # 用户问题
    documents: List[Document]        # 检索到的文档列表
    generation: str                  # 生成的答案
    run_web_search: str              # 是否需要web搜索 ("Yes" or "No")
    retrieval_k: int                 # 检索数量


# =============================
# 节点函数（保留原始代码的所有逻辑和提示词）
# =============================

def retrieve(state):
    """检索节点：从向量库检索"""
    question = state["question"]
    k = state.get("retrieval_k", 15)  # 从 state 获取 k 值，默认 15

    retriever = get_retriever(k=k)
    if retriever is None:
        return {"documents": [], "question": question}

    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def generate(state):
    """Generate answer using vLLM model"""
    print("[Generate] Generating answer...")
    question, documents = state["question"], state["documents"]
    try:
        prompt = PromptTemplate(template="""Based on the following context, please answer the question.
            Context: {context}
            Question: {question}
            Answer:""", input_variables=["context", "question"])
        llm = ChatOpenAI(
            model=VLLM_MODEL_NAME,
            openai_api_key=VLLM_API_KEY,
            openai_api_base=VLLM_API_BASE,
            temperature=0,
            max_tokens=1000
        )
        context = "\n\n".join(doc.page_content for doc in documents)

        # Create and run chain
        rag_chain = (
            {"context": lambda x: context, "question": lambda x: question}
            | prompt
            | llm
            | StrOutputParser()
        )

        generation = rag_chain.invoke({})

        return {
            "documents": documents,
            "question": question,
            "generation": generation
        }

    except Exception as e:
        error_msg = f"Error in generate function: {str(e)}"
        print(error_msg)
        return {"documents": documents, "question": question,
                "generation": "Sorry, I encountered an error while generating the response."}


def grade_documents(state):
    """Determines whether the retrieved documents are relevant."""
    print("[Grade documents] Checking relevance...")
    question = state["question"]
    documents = state["documents"]

    llm = ChatOpenAI(
        model=VLLM_MODEL_NAME,
        openai_api_key=VLLM_API_KEY,
        openai_api_base=VLLM_API_BASE,
        temperature=0,
        max_tokens=1000
    )

    prompt = PromptTemplate(template="""You are grading the relevance of a retrieved document to a user question.
        Return ONLY a JSON object with a "score" field that is either "yes" or "no".
        Do not include any other text or explanation.

        Document: {context}
        Question: {question}

        Rules:
        - Check for related keywords or semantic meaning
        - Use lenient grading to only filter clear mismatches
        - Return exactly like this example: {{"score": "yes"}} or {{"score": "no"}}""",
        input_variables=["context", "question"])

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    filtered_docs = []
    search = "No"

    for d in documents:
        try:
            response = chain.invoke({"question": question, "context": d.page_content})
            json_match = re.search(r'\{.*\}', response)
            if json_match:
                response = json_match.group()

            score = json.loads(response)

            if score.get("score") == "yes":
                print("[Grade documents] Document is relevant.")
                filtered_docs.append(d)
            else:
                print("[Grade documents] Document is not relevant.")
                search = "Yes"

        except Exception as e:
            print(f"Error grading document: {str(e)}")
            # On error, keep the document to be safe
            filtered_docs.append(d)
            continue

    return {"documents": filtered_docs, "question": question, "run_web_search": search}


def transform_query(state):
    """Transform the query to produce a better question."""
    print("[Transform query] Transforming query...")
    question = state["question"]
    documents = state["documents"]

    # Create a prompt template
    prompt = PromptTemplate(
        template="""Generate a search-optimized version of this question by
        analyzing its core semantic meaning and intent.
        \n ------- \n
        {question}
        \n ------- \n
        Return only the improved question with no additional text:""",
        input_variables=["question"],
    )

    # Use vLLM
    llm = ChatOpenAI(
        model=VLLM_MODEL_NAME,
        openai_api_key=VLLM_API_KEY,
        openai_api_base=VLLM_API_BASE,
        temperature=0,
        max_tokens=1000
    )

    # Prompt
    chain = prompt | llm | StrOutputParser()
    better_question = chain.invoke({"question": question})

    return {
        "documents": documents, "question": better_question
    }


def web_search(state):
    """Web search based on the re-phrased question using Tavily API."""
    print("[Web search] Executing web search...")
    question = state["question"]
    documents = state["documents"]

    try:
        # Get cached search tool
        tool = get_web_search_tool()
        if tool is None:
            print("[Web search] Tavily API key not provided - skipping web search")
            return {"documents": documents, "question": question}

        # Execute search with retry logic
        print("[Web search] Executing search query...")
        try:
            from tenacity import retry, stop_after_attempt, wait_exponential

            @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
            def execute_tavily_search(tool, query):
                # New API uses invoke(query) directly, old API uses invoke({"query": query})
                try:
                    return tool.invoke(query)
                except (TypeError, KeyError):
                    return tool.invoke({"query": query})

            search_results = execute_tavily_search(tool, question)
        except Exception as search_error:
            print(f"[Web search] Search failed after retries: {str(search_error)}")
            return {"documents": documents, "question": question}

        if not search_results:
            print("[Web search] No search results found")
            return {"documents": documents, "question": question}

        # Process results - handle both string and list formats
        print("[Web search] Processing search results...")
        web_results = []

        # New API returns a string, old API returns a list of dicts
        if isinstance(search_results, str):
            # New API: just use the string directly
            web_results.append(search_results)
        elif isinstance(search_results, list):
            # Old API: extract title and content from each result
            for result in search_results:
                if isinstance(result, dict):
                    content = (
                        f"Title: {result.get('title', 'No title')}\n"
                        f"Content: {result.get('content', 'No content')}\n"
                    )
                    web_results.append(content)
                else:
                    # Fallback: convert to string
                    web_results.append(str(result))
        else:
            # Unknown format: convert to string
            web_results.append(str(search_results))

        # Create document from results
        web_document = Document(
            page_content="\n\n".join(web_results),
            metadata={
                "source": "tavily_search",
                "query": question,
                "result_count": len(web_results)
            }
        )
        documents.append(web_document)

        print(f"[Web search] Successfully added {len(web_results)} search results")

    except Exception as error:
        error_msg = f"Web search error: {str(error)}"
        print(error_msg)

    return {"documents": documents, "question": question}


def decide_to_generate(state):
    print("[Decide to generate] Making decision...")
    search = state.get("run_web_search", "No")

    if search == "Yes":
        print("[Decide to generate] Decision: transform query and run web search")
        return "transform_query"
    else:
        print("[Decide to generate] Decision: generate answer directly")
        return "generate"


# =============================
# 工作流构建（保留原始结构）
# =============================

def build_crag2_workflow():
    workflow = StateGraph(GraphState)

    # Define the nodes by langgraph
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search", web_search)

    # Build graph
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "web_search")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()

def save_workflow_graph(output_path: str):
    """保存工作流图到文件"""
    from pathlib import Path

    print("=" * 60)
    print("[Workflow Graph] Saving workflow graph...")

    # 如果传入的是目录，自动添加文件名
    output_path = Path(output_path)
    if output_path.is_dir() or not output_path.suffix:
        output_path = output_path / "crag2_workflow.png"

    # 确保目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        workflow = build_crag2_workflow()

        # 尝试使用 draw_mermaid_png
        try:
            png_data = workflow.get_graph().draw_mermaid_png()
            with open(output_path, "wb") as f:
                f.write(png_data)
            print(f"[Workflow Graph] Saved workflow graph to {output_path} using draw_mermaid_png")
            return True

        except Exception as e:
            print(f"[Workflow Graph] Failed to save using draw_mermaid_png: {e}")

        # 尝试使用 draw_png
        try:
            png_data = workflow.get_graph().draw_png()
            with open(output_path, "wb") as f:
                f.write(png_data)
            print(f"[Workflow Graph] Saved workflow graph to {output_path} using draw_png")
            return True
        except Exception as e:
            print(f"[Workflow Graph] Failed to save using draw_png: {e}")

        print("[Workflow Graph] All methods failed to save workflow graph")
        return False

    except Exception as e:
        print(f"[Workflow Graph] Error building workflow: {e}")
        return False

# =============================
# 主接口函数
# =============================

def run_c_rag2(question: str, retrieval_k: int = 15) -> Dict[str, Any]:
    """
    运行 CRag2 查询

    Args:
        question: 用户问题
        retrieval_k: 检索数量（默认 15）

    Returns:
        包含 answer 和 metadata 的字典
    """
    app = build_crag2_workflow()

    inputs = {
        "question": question,
        "retrieval_k": retrieval_k
    }

    result = app.invoke(
        inputs,
        config={
            "callbacks": [monitor],
            "recursion_limit": 100
        }
    )

    return {
        "answer": result.get("generation", ""),
        "metadata": {
            "used_web_search": result.get("run_web_search", "No") == "Yes",
            "retrieval_k": retrieval_k
        },
        "documents": result.get("documents", [])
    }


# =============================
# 命令行测试
# =============================

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
    else:
        q = "What are the latest developments in AI?"

    print(f"Question: {q}")
    result = run_c_rag2(q)
    print(f"Answer: {result['answer']}")
    print(f"Metadata: {result['metadata']}")
