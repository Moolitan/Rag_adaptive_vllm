import time
import json
import uuid
import re
import math
import threading
from contextvars import ContextVar
from typing import TypedDict, List, Dict, Any, Tuple, Optional
from pathlib import Path
from functools import wraps
from collections import Counter

import numpy as np
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.graph import StateGraph, START, END

# =============================
# 状态定义
# =============================

class Hop2RagState(TypedDict, total=False):
    """Hop2Rag状态定义"""
    # 核心输入输出
    question: str                         # 用户问题
    generation: str                       # 生成的答案
    documents: List[Document]             # 当前文档列表

    # 数据源信息
    data_source: str                       # 使用的数据源
    used_web_search: bool                  # 是否使用了web搜索

    # 文档评分
    document_scores: List[Dict[str, Any]]  # 文档评分列表
    graded_relevant_docs: int              # 相关文档数量
    has_irrelevant_docs: bool              # 是否有不相关文档

    # 多跳控制
    current_hop: int                       # 当前跳数 (0, 1, 2, ...)
    max_hops: int                          # 最大跳数
    hop_k: int                             # 每跳检索K值

    # 跳数历史追踪
    hop_queries: List[str]                 # 每跳查询
    hop_documents: List[List[Document]]    # 每跳检索的文档
    hop_evidence: List[str]                # 每跳提取的证据

    # 中间推理
    intermediate_answers: List[str]        # 每跳的部分答案
    all_graded_documents: List[Document]   # 跨跳累积的相关文档

    # Supporting facts
    supporting_facts: List                 # HotpotQA格式的支撑事实

    # 最终输出
    final_answer: str                      # 最终答案
    metadata: Dict[str, Any]               # 元数据

    # 自定义检索配置
    custom_retriever_config: Dict[str, str]

    # 插桩数据
    _timings_ms: Dict[str, float]          # 节点耗时
    _edge_timings_ms: Dict[str, float]     # 边耗时


# =============================
# 插桩工具
# =============================

class PerformanceTracker:
    """性能追踪器:管理 timing 和 JSONL 输出"""

    def __init__(self, output_dir: str = None, filename: str = "hop2rag_timings.jsonl"):
        self.output_dir = Path(output_dir) if output_dir else None
        self.filename = filename
        self.file_handle = None
        self.jsonl_path = None
        self._enabled = False
        # LLM prompt 记录
        self._prompt_log_path = None
        self._prompt_file_handle = None

    def enable(self, output_dir: str, filename: str = "hop2rag_timings.jsonl"):
        """启用性能追踪并设置输出目录"""
        self.close()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.filename = filename
        self.jsonl_path = self.output_dir / self.filename
        self.file_handle = open(self.jsonl_path, "a", buffering=1<<20)
        # 初始化 prompt 日志
        self._prompt_log_path = self.output_dir / "llm_prompts.jsonl"
        self._prompt_file_handle = open(self._prompt_log_path, "a", buffering=1<<20)
        self._enabled = True

    def disable(self):
        """禁用性能追踪"""
        self.close()
        self._enabled = False

    def log_request(self, record: Dict[str, Any]):
        """追加写入一条 JSONL 记录"""
        if self._enabled and self.file_handle:
            self.file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.file_handle.flush()

    def log_llm_prompt(self, record: Dict[str, Any]):
        """记录一次 LLM 调用的 prompt 详情"""
        if self._enabled and self._prompt_file_handle:
            self._prompt_file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._prompt_file_handle.flush()

    def clear_log(self):
        """清空日志文件"""
        if self.jsonl_path and self.jsonl_path.exists():
            self.close()
            self.jsonl_path.unlink()
            if self._prompt_log_path and self._prompt_log_path.exists():
                self._prompt_log_path.unlink()
            if self._enabled:
                self.file_handle = open(self.jsonl_path, "a", buffering=1<<20)
                self._prompt_file_handle = open(self._prompt_log_path, "a", buffering=1<<20)

    def close(self):
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
        if self._prompt_file_handle:
            self._prompt_file_handle.close()
            self._prompt_file_handle = None

    def __del__(self):
        self.close()


_tracker = PerformanceTracker()


def enable_instrumentation(output_dir: str, filename: str = "hop2rag_timings.jsonl"):
    """启用性能追踪"""
    _tracker.enable(output_dir, filename)


def disable_instrumentation():
    """禁用性能追踪"""
    _tracker.disable()


def clear_instrumentation_log():
    """清空性能日志"""
    _tracker.clear_log()


def log_llm_call(request_id: str, node_name: str, hop: int, prompt: str, prompt_tokens: int = 0):
    """
    记录一次 LLM 调用的详细信息（用于 KV Cache 分析）

    Args:
        request_id: 请求 ID
        node_name: 节点名称 (decompose, extract_clues, decide, generate)
        hop: 当前 hop 数
        prompt: 完整的 prompt 文本
        prompt_tokens: prompt token 数量（如果已知）
    """
    record = {
        "request_id": request_id,
        "node_name": node_name,
        "hop": hop,
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "timestamp": time.time(),
    }
    _tracker.log_llm_prompt(record)


def instrument_node(node_name: str):
    """节点插桩装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(state: Dict[str, Any], config=None):
            if "_timings_ms" not in state:
                state["_timings_ms"] = {}

            start = time.perf_counter()
            result = func(state, config) if config is not None else func(state)
            elapsed_ms = (time.perf_counter() - start) * 1000

            state["_timings_ms"][node_name] = state["_timings_ms"].get(node_name, 0) + elapsed_ms

            if isinstance(result, dict):
                result["_timings_ms"] = state["_timings_ms"]

            return result
        return wrapper
    return decorator


def instrument_edge(edge_name: str):
    """边插桩装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(state: Dict[str, Any]):
            if "_edge_timings_ms" not in state:
                state["_edge_timings_ms"] = {}

            start = time.perf_counter()
            result = func(state)
            elapsed_ms = (time.perf_counter() - start) * 1000

            state["_edge_timings_ms"][edge_name] = state["_edge_timings_ms"].get(edge_name, 0) + elapsed_ms

            return result
        return wrapper
    return decorator


# =============================
# Usage Aggregator（按 request 级汇总 LLM usage）
# =============================

# 使用 ContextVar 实现并发安全的 request_id 绑定
_current_request_id: ContextVar[Optional[str]] = ContextVar("current_request_id", default=None)


class UsageAggregator:
    """
    按 request_id 聚合所有 LLM 调用的 token usage。
    并发安全：使用 threading.Lock 保护内部字典。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._usage_data: Dict[str, Dict[str, Any]] = {}

    def init_request(self, request_id: str):
        """初始化一个请求的 usage 数据"""
        with self._lock:
            self._usage_data[request_id] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "call_count": 0,
                "has_real_usage": False,  # 是否有真实 usage 数据
            }

    def add_usage(self, request_id: str, prompt_tokens: int, completion_tokens: int, from_real_usage: bool = True):
        """累加一次 LLM 调用的 usage"""
        with self._lock:
            if request_id not in self._usage_data:
                self.init_request(request_id)
            data = self._usage_data[request_id]
            data["prompt_tokens"] += prompt_tokens
            data["completion_tokens"] += completion_tokens
            data["total_tokens"] += prompt_tokens + completion_tokens
            data["call_count"] += 1
            if from_real_usage:
                data["has_real_usage"] = True

    def get_aggregated_usage(self, request_id: str) -> Dict[str, Any]:
        """获取聚合后的 usage，包含 token_source 字段"""
        with self._lock:
            if request_id not in self._usage_data:
                return {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "token_source": "unknown",
                    "llm_call_count": 0,
                }
            data = self._usage_data[request_id]
            # 判断 token_source
            if data["has_real_usage"]:
                token_source = "usage"
            elif data["call_count"] > 0:
                token_source = "tokenizer_estimate"
            else:
                token_source = "unknown"
            return {
                "prompt_tokens": data["prompt_tokens"],
                "completion_tokens": data["completion_tokens"],
                "total_tokens": data["total_tokens"],
                "token_source": token_source,
                "llm_call_count": data["call_count"],
            }

    def cleanup_request(self, request_id: str):
        """清理请求的 usage 数据（请求完成后调用）"""
        with self._lock:
            self._usage_data.pop(request_id, None)


# 全局 Usage Aggregator 实例
_usage_aggregator = UsageAggregator()


class UsageCollectorCallback(BaseCallbackHandler):
    """
    LangChain Callback Handler：在每次 LLM 结束时捕获 usage 并累加到当前 request_id。
    支持从多种位置提取 usage：response.usage / generation_info / response_metadata。
    """

    def __init__(self, aggregator: UsageAggregator):
        super().__init__()
        self.aggregator = aggregator

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """LLM 调用结束时捕获 usage"""
        request_id = _current_request_id.get()
        if not request_id:
            return

        prompt_tokens = 0
        completion_tokens = 0
        from_real_usage = False

        # 尝试从多个位置提取 usage
        # 1. 从 llm_output 提取（OpenAI 风格）
        if response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
            if token_usage:
                prompt_tokens = token_usage.get("prompt_tokens", 0)
                completion_tokens = token_usage.get("completion_tokens", 0)
                from_real_usage = True

            # vLLM / OpenAI 可能使用 usage 字段
            if not from_real_usage:
                usage = response.llm_output.get("usage", {})
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    from_real_usage = True

        # 2. 从 generations 中的 generation_info / message.response_metadata 提取
        if not from_real_usage and response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    # 尝试 generation_info
                    gen_info = getattr(gen, "generation_info", {}) or {}
                    usage = gen_info.get("usage", {}) or gen_info.get("token_usage", {})
                    if usage:
                        prompt_tokens += usage.get("prompt_tokens", 0)
                        completion_tokens += usage.get("completion_tokens", 0)
                        from_real_usage = True
                        break

                    # 尝试 message.response_metadata（ChatOpenAI 返回的 AIMessage）
                    msg = getattr(gen, "message", None)
                    if msg:
                        resp_meta = getattr(msg, "response_metadata", {}) or {}
                        usage = resp_meta.get("token_usage", {}) or resp_meta.get("usage", {})
                        if usage:
                            prompt_tokens += usage.get("prompt_tokens", 0)
                            completion_tokens += usage.get("completion_tokens", 0)
                            from_real_usage = True
                            break

                        # additional_kwargs
                        add_kwargs = getattr(msg, "additional_kwargs", {}) or {}
                        usage = add_kwargs.get("usage", {})
                        if usage:
                            prompt_tokens += usage.get("prompt_tokens", 0)
                            completion_tokens += usage.get("completion_tokens", 0)
                            from_real_usage = True
                            break
                if from_real_usage:
                    break

        # 累加到 aggregator
        if prompt_tokens > 0 or completion_tokens > 0:
            self.aggregator.add_usage(request_id, prompt_tokens, completion_tokens, from_real_usage)


# 全局 callback 实例
_usage_callback = UsageCollectorCallback(_usage_aggregator)


def get_usage_callbacks() -> list:
    """获取包含 usage 收集的 callbacks 列表"""
    return [_usage_callback]


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

QUESTION_DECOMPOSER_PROMPT = """You are a question decomposition expert for multi-hop reasoning tasks.

Current hop: {current_hop}
Original question: {original_question}
Evidence so far: {evidence}
Previous queries: {previous_queries}

Instructions:
- Generate a NEW search query that is DIFFERENT from all previous queries
- Hop 0 (initial): Extract the first entity/concept to search for
- Hop 1+: Based on evidence, determine what to search next
- For bridge questions: Find intermediate entity, then search for final answer
- For comparison questions: If hop 0 searched entity A, hop 1 must search entity B
- If you believe we have sufficient information, output "DONE" as the query

Return JSON: {{"query": "search query for this hop OR 'DONE' if sufficient", "reasoning": "why this query", "target_entity": "the main entity being searched"}}"""

CLUE_EXTRACTOR_PROMPT = """Extract key entities/clues from retrieved documents for multi-hop reasoning.

Original question: {question}
Current hop: {current_hop}
Retrieved documents:
{documents}

Instructions:
- Extract the key entity, fact, or concept that answers the current sub-question
- For bridge questions: Extract intermediate entity (e.g., person name, place)
- For comparison questions: Extract specific attribute to compare
- Be precise - this will guide the next hop's retrieval

Return JSON: {{"clues": ["entity1", "entity2"], "summary": "brief summary"}}"""

HOP_DECISION_PROMPT = """Determine if another retrieval hop is needed.

Original question: {question}
Current hop: {current_hop}
Max hops: {max_hops}
Evidence collected:
{evidence}
Documents retrieved: {doc_count}

Instructions:
- Answer "yes" if you have sufficient information to answer the question
- Answer "no" if you need more information AND haven't hit max_hops
- Consider: Do we have both pieces of info for bridge/comparison questions?

Return JSON: {{"decision": "yes" or "no", "reasoning": "explanation"}}"""

MULTI_HOP_RAG_PROMPT = """Answer a multi-hop question using evidence from multiple retrieval steps.

Original question: {question}

=== Collected Evidence ===
{evidence}

=== Supporting Documents ===
{context}

Instructions:
1. Synthesize information from the evidence across ALL hops
2. For comparison questions: Compare the specific attributes found
3. For bridge questions: Connect the intermediate entity to the final answer
4. Give a direct, concise answer grounded in the evidence

Answer:"""


# =============================
# BM25 Reranker (修复后的实现)
# =============================

class BM25Reranker:
    """基于BM25的文档重排序器"""

    def __init__(self, k1: float = 1.5, b: float = 0.75, threshold: float = 0.0):
        self.k1 = k1
        self.b = b
        self.threshold = threshold

    def _tokenize(self, text: str) -> List[str]:
        """简单分词"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return [w for w in text.split() if len(w) > 1]

    def score_documents(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """使用BM25对文档评分"""
        if not documents:
            return []

        query_tokens = self._tokenize(query)
        doc_tokens_list = [self._tokenize(d.page_content[:1000]) for d in documents]

        doc_freq = Counter()
        for doc_tokens in doc_tokens_list:
            unique_tokens = set(doc_tokens)
            for token in unique_tokens:
                doc_freq[token] += 1

        avg_dl = sum(len(dt) for dt in doc_tokens_list) / len(doc_tokens_list) if doc_tokens_list else 1
        n_docs = len(documents)

        scores = []
        for i, (doc, doc_tokens) in enumerate(zip(documents, doc_tokens_list)):
            score = 0.0
            dl = len(doc_tokens)
            token_counts = Counter(doc_tokens)

            for q_token in query_tokens:
                if q_token in token_counts:
                    df = doc_freq.get(q_token, 0)
                    idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
                    tf = token_counts[q_token]
                    tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / avg_dl))
                    score += idf * tf_norm

            scores.append((doc, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


_reranker = None


def get_reranker(threshold: float = 0.0) -> BM25Reranker:
    """获取reranker实例"""
    global _reranker
    if _reranker is None:
        _reranker = BM25Reranker(threshold=threshold)
    return _reranker


# =============================
# Sentence Selector (使用BM25)
# =============================

class SentenceSelector:
    """基于BM25的句子级supporting facts选择器"""

    def __init__(self, top_m: int = 3):
        self.top_m = top_m
        self.reranker = BM25Reranker()

    def extract_supporting_facts(
        self,
        question: str,
        answer: str,
        documents: List[Document]
    ) -> List[Tuple[str, int]]:
        """提取supporting facts"""
        query = f"{question} {answer}"

        all_sentences = []
        for doc in documents:
            doc_title = (
                doc.metadata.get("title")
                or doc.metadata.get("page_title")
                or doc.metadata.get("wiki_title")
                or doc.metadata.get("source")
                or doc.metadata.get("doc_id")
                or doc.metadata.get("id")
            )
            if not doc_title or (isinstance(doc_title, str) and not doc_title.strip()):
                content_hash = hex(abs(hash(doc.page_content)))[2:10]
                doc_title = f"Doc-{content_hash}"

            sentences = self._split_sentences(doc.page_content)
            for sent_idx, sent_text in enumerate(sentences):
                if len(sent_text.strip()) > 10:
                    all_sentences.append((doc_title, sent_idx, sent_text))

        if not all_sentences:
            return []

        pseudo_docs = [Document(page_content=s[2]) for s in all_sentences]
        scored = self.reranker.score_documents(query, pseudo_docs)

        doc_to_info = {id(pd): (all_sentences[i][0], all_sentences[i][1])
                       for i, pd in enumerate(pseudo_docs)}

        result = []
        for doc, score in scored[:self.top_m]:
            if id(doc) in doc_to_info:
                title, idx = doc_to_info[id(doc)]
                result.append((title, idx))

        return result

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


_sentence_selector = None


def get_sentence_selector(top_m: int = 3) -> SentenceSelector:
    """获取sentence selector实例"""
    global _sentence_selector
    if _sentence_selector is None:
        _sentence_selector = SentenceSelector(top_m=top_m)
    return _sentence_selector


# =============================
# 向量库配置
# =============================
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

_custom_retrievers = {}


def get_custom_retriever(persist_dir: str, collection_name: str, k: int = 10):
    """获取自定义retriever"""
    cache_key = f"{persist_dir}:{collection_name}:{k}"
    if cache_key not in _custom_retrievers:
        if not persist_dir:
            raise ValueError("persist_dir is required")
        if not collection_name:
            raise ValueError("collection_name is required")

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
        _custom_retrievers[cache_key] = vectorstore.as_retriever(search_kwargs={"k": k})

    return _custom_retrievers[cache_key]


# =============================
# 链构建
# =============================

_question_decomposer = None
_clue_extractor = None
_hop_decision_chain = None
_multi_hop_rag_chain = None


def get_question_decomposer():
    """获取问题分解链"""
    global _question_decomposer
    if _question_decomposer is None:
        prompt = PromptTemplate(
            template=QUESTION_DECOMPOSER_PROMPT,
            input_variables=["current_hop", "original_question", "evidence", "previous_queries"]
        )
        _question_decomposer = prompt | get_llm(json_mode=True) | JsonOutputParser()
    return _question_decomposer


def get_clue_extractor():
    """获取线索提取链"""
    global _clue_extractor
    if _clue_extractor is None:
        prompt = PromptTemplate(
            template=CLUE_EXTRACTOR_PROMPT,
            input_variables=["question", "current_hop", "documents"]
        )
        _clue_extractor = prompt | get_llm(json_mode=True) | JsonOutputParser()
    return _clue_extractor


def get_hop_decision_chain():
    """获取跳数决策链"""
    global _hop_decision_chain
    if _hop_decision_chain is None:
        prompt = PromptTemplate(
            template=HOP_DECISION_PROMPT,
            input_variables=["question", "current_hop", "max_hops", "evidence", "doc_count"]
        )
        _hop_decision_chain = prompt | get_llm(json_mode=True) | JsonOutputParser()
    return _hop_decision_chain


def get_multi_hop_rag_chain():
    """获取多跳RAG生成链"""
    global _multi_hop_rag_chain
    if _multi_hop_rag_chain is None:
        prompt = PromptTemplate(
            template=MULTI_HOP_RAG_PROMPT,
            input_variables=["question", "evidence", "context"]
        )
        _multi_hop_rag_chain = prompt | get_llm() | StrOutputParser()
    return _multi_hop_rag_chain


# =============================
# 节点函数(插桩版)
# =============================

@instrument_node("initialize_multi_hop")
def initialize_node(state: Hop2RagState) -> Dict[str, Any]:
    """初始化多跳状态"""
    cfg = state.get("custom_retriever_config", {})
    hop_k = int(cfg.get("k", 10))
    max_hops = int(cfg.get("max_hops", 5))

    return {
        "current_hop": 0,
        "max_hops": max_hops,
        "hop_k": hop_k,
        "hop_queries": [],
        "hop_documents": [],
        "hop_evidence": [],
        "intermediate_answers": [],
        "all_graded_documents": [],
        "metadata": state.get("metadata", {}),
        "_timings_ms": {},
        "_edge_timings_ms": {}
    }


@instrument_node("decompose_question")
def decompose_node(state: Hop2RagState) -> Dict[str, Any]:
    """问题分解节点"""
    current_hop = state.get("current_hop", 0)
    original_question = state["question"]
    evidence = state.get("hop_evidence", [])
    hop_queries = state.get("hop_queries", [])
    metadata = state.get("metadata", {})

    evidence_text = "\n".join([f"Hop {i}: {ev}" for i, ev in enumerate(evidence)]) if evidence else "None"
    previous_queries_text = ", ".join([f'"{q}"' for q in hop_queries]) if hop_queries else "None"

    # 构建 prompt 用于记录
    prompt_text = QUESTION_DECOMPOSER_PROMPT.format(
        current_hop=current_hop,
        original_question=original_question,
        evidence=evidence_text,
        previous_queries=previous_queries_text
    )

    # 记录 LLM 调用（如果启用了 instrumentation）
    request_id = _current_request_id.get()
    if request_id:
        log_llm_call(request_id, "decompose", current_hop, prompt_text)

    decomposer = get_question_decomposer()
    result = decomposer.invoke({
        "current_hop": current_hop,
        "original_question": original_question,
        "evidence": evidence_text,
        "previous_queries": previous_queries_text
    })

    raw_query = result.get("query", "")
    next_query = raw_query.strip() if isinstance(raw_query, str) else ""

    metadata_updated = metadata.copy()

    # 检查是否LLM认为信息已足够
    if next_query.upper() == "DONE":
        metadata_updated[f"hop_{current_hop}_decompose_done"] = True
        metadata_updated["early_stop_signal"] = True
        next_query = original_question

    if not next_query:
        next_query = original_question.strip()
        metadata_updated[f"hop_{current_hop}_decompose_fallback"] = True

    hop_queries_updated = hop_queries + [next_query]

    return {"hop_queries": hop_queries_updated, "metadata": metadata_updated}


@instrument_node("retrieve_hop_documents")
def retrieve_hop_node(state: Hop2RagState) -> Dict[str, Any]:
    """检索当前跳的文档"""
    current_hop = state.get("current_hop", 0)
    hop_queries = state.get("hop_queries", [])
    hop_k = state.get("hop_k", 10)
    metadata = state.get("metadata", {})

    if current_hop < len(hop_queries):
        query = hop_queries[current_hop]
    else:
        query = state["question"]

    cfg = state.get("custom_retriever_config", {})
    custom_retriever = get_custom_retriever(
        persist_dir=cfg.get("persist_dir", ""),
        collection_name=cfg.get("collection_name", "hotpot_fullwiki"),
        k=hop_k
    )

    docs = custom_retriever.invoke(query)

    metadata_updated = metadata.copy()
    metadata_updated[f"hop_{current_hop}_retrieve_query"] = query

    return {
        "documents": docs,
        "data_source": "custom_vectorstore",
        "metadata": metadata_updated
    }


@instrument_node("rerank_documents")
def rerank_node(state: Hop2RagState) -> Dict[str, Any]:
    """使用BM25重排序文档"""
    docs = state.get("documents", [])
    if not docs:
        return {"documents": [], "document_scores": []}

    current_hop = state.get("current_hop", 0)
    hop_queries = state.get("hop_queries", [])
    original_question = state["question"]
    metadata = state.get("metadata", {})

    if current_hop < len(hop_queries):
        hop_query = hop_queries[current_hop]
    else:
        hop_query = original_question

    rerank_query = f"{original_question} {hop_query}"

    reranker = get_reranker()
    scored_docs = reranker.score_documents(rerank_query, docs)

    # 保留top文档
    top_k = min(5, len(scored_docs))
    kept_docs = [doc for doc, score in scored_docs[:top_k]]

    document_scores = [{"document": doc, "raw_score": score} for doc, score in scored_docs[:top_k]]

    metadata_updated = metadata.copy()
    metadata_updated[f"hop_{current_hop}_rerank_query"] = rerank_query

    return {
        "documents": kept_docs,
        "document_scores": document_scores,
        "graded_relevant_docs": len(kept_docs),
        "metadata": metadata_updated
    }


@instrument_node("accumulate_graded_documents")
def accumulate_node(state: Hop2RagState) -> Dict[str, Any]:
    """累积已评分的相关文档"""
    current_hop = state.get("current_hop", 0)
    current_docs = state.get("documents", [])
    all_graded = state.get("all_graded_documents", [])
    hop_documents = state.get("hop_documents", [])

    # 去重累积，每hop最多加3篇
    existing_contents = {d.page_content for d in all_graded}
    new_docs = [d for d in current_docs[:3] if d.page_content not in existing_contents]
    all_graded_updated = all_graded + new_docs

    hop_documents_updated = hop_documents + [current_docs]

    return {
        "all_graded_documents": all_graded_updated,
        "hop_documents": hop_documents_updated
    }


@instrument_node("extract_clues")
def extract_clues_node(state: Hop2RagState) -> Dict[str, Any]:
    """线索提取节点"""
    current_hop = state.get("current_hop", 0)
    question = state["question"]
    documents = state.get("documents", [])
    hop_evidence = state.get("hop_evidence", [])
    intermediate_answers = state.get("intermediate_answers", [])

    docs_text = "\n\n".join([
        f"Doc {i+1}: {d.page_content[:300]}"
        for i, d in enumerate(documents[:10])
    ])

    # 构建 prompt 用于记录
    prompt_text = CLUE_EXTRACTOR_PROMPT.format(
        question=question,
        current_hop=current_hop,
        documents=docs_text
    )

    # 记录 LLM 调用
    request_id = _current_request_id.get()
    if request_id:
        log_llm_call(request_id, "extract_clues", current_hop, prompt_text)

    extractor = get_clue_extractor()
    result = extractor.invoke({
        "question": question,
        "current_hop": current_hop,
        "documents": docs_text
    })

    clues = result.get("clues", [])
    summary = result.get("summary", "")
    evidence_text = f"{', '.join(clues)} - {summary}" if clues else summary

    hop_evidence_updated = hop_evidence + [evidence_text]
    intermediate_answers_updated = intermediate_answers + [summary]

    return {
        "hop_evidence": hop_evidence_updated,
        "intermediate_answers": intermediate_answers_updated
    }


@instrument_node("decide_next_hop")
def decide_node(state: Hop2RagState) -> Dict[str, Any]:
    """跳数决策节点"""
    current_hop = state.get("current_hop", 0)
    max_hops = state.get("max_hops", 5)
    question = state["question"]
    hop_evidence = state.get("hop_evidence", [])
    all_docs = state.get("all_graded_documents", [])
    metadata = state.get("metadata", {})

    # 检查decompose是否发出了提前停止信号
    early_stop = metadata.get("early_stop_signal", False)

    evidence_text = "\n".join([f"Hop {i}: {ev}" for i, ev in enumerate(hop_evidence)])

    # 构建 prompt 用于记录
    prompt_text = HOP_DECISION_PROMPT.format(
        question=question,
        current_hop=current_hop,
        max_hops=max_hops,
        evidence=evidence_text,
        doc_count=len(all_docs)
    )

    # 记录 LLM 调用
    request_id = _current_request_id.get()
    if request_id:
        log_llm_call(request_id, "decide", current_hop, prompt_text)

    decision_chain = get_hop_decision_chain()
    result = decision_chain.invoke({
        "question": question,
        "current_hop": current_hop,
        "max_hops": max_hops,
        "evidence": evidence_text,
        "doc_count": len(all_docs)
    })

    raw_decision_raw = result.get("decision", "no")
    reasoning = result.get("reasoning", "")

    if isinstance(raw_decision_raw, str):
        raw_decision = raw_decision_raw.strip().lower()
    else:
        raw_decision = "no"

    if raw_decision not in ("yes", "no"):
        raw_decision = "no"

    # 决策逻辑
    if current_hop >= max_hops - 1:
        decision = "yes"  # 强制终止
    elif early_stop or raw_decision == "yes":
        decision = "yes"
    else:
        decision = "no"

    metadata_updated = metadata.copy()
    metadata_updated[f"hop_{current_hop}_decision"] = decision
    metadata_updated[f"hop_{current_hop}_decision_raw"] = raw_decision_raw
    metadata_updated[f"hop_{current_hop}_reasoning"] = reasoning
    if "early_stop_signal" in metadata_updated:
        del metadata_updated["early_stop_signal"]

    next_hop = current_hop + 1

    return {
        "current_hop": next_hop,
        "metadata": metadata_updated
    }


@instrument_node("generate_multi_hop_final")
def generate_final_node(state: Hop2RagState) -> Dict[str, Any]:
    """多跳最终生成节点"""
    question = state["question"]
    hop_evidence = state.get("hop_evidence", [])
    hop_queries = state.get("hop_queries", [])
    all_graded = state.get("all_graded_documents", [])
    hop_documents = state.get("hop_documents", [])
    current_hop = state.get("current_hop", 0)

    # 构建证据文本
    evidence_parts = []
    for i, ev in enumerate(hop_evidence):
        query = hop_queries[i] if i < len(hop_queries) else ""
        evidence_parts.append(f"Hop {i} (Query: {query}):\n{ev}")
    evidence_text = "\n\n".join(evidence_parts)

    # 构建支撑文档（优先最后一跳 + 累积的top文档）
    final_docs = []
    if hop_documents:
        final_docs.extend(hop_documents[-1][:3])
    existing = {d.page_content for d in final_docs}
    for d in all_graded:
        if d.page_content not in existing and len(final_docs) < 6:
            final_docs.append(d)
            existing.add(d.page_content)

    context = "\n\n".join([
        f"[Doc {i+1}] {d.page_content[:500]}"
        for i, d in enumerate(final_docs)
    ])

    # 构建 prompt 用于记录
    prompt_text = MULTI_HOP_RAG_PROMPT.format(
        question=question,
        evidence=evidence_text,
        context=context
    )

    # 记录 LLM 调用
    request_id = _current_request_id.get()
    if request_id:
        log_llm_call(request_id, "generate", current_hop, prompt_text)

    rag_chain = get_multi_hop_rag_chain()
    result = rag_chain.invoke({
        "question": question,
        "evidence": evidence_text,
        "context": context
    })

    return {
        "generation": result,
        "documents": final_docs
    }


@instrument_node("extract_supporting_facts_fast")
def extract_sp_fast_node(state: Hop2RagState) -> Dict[str, Any]:
    """使用BM25快速提取supporting facts"""
    question = state["question"]
    answer = state.get("generation", "")
    documents = state.get("documents", [])

    if not documents:
        return {"supporting_facts": []}

    selector = get_sentence_selector(top_m=3)
    sp_facts = selector.extract_supporting_facts(question, answer, documents)

    supporting_facts = [[title, idx] for title, idx in sp_facts]

    return {"supporting_facts": supporting_facts}


@instrument_node("finalize")
def finalize_node(state: Hop2RagState) -> Dict[str, Any]:
    """最终化输出"""
    total_hops = state.get("current_hop", 0)
    hop_queries = state.get("hop_queries", [])

    existing_metadata = state.get("metadata", {})

    final_metadata = {
        **existing_metadata,
        "data_source": state.get("data_source"),
        "used_web_search": state.get("used_web_search", False),
        "graded_relevant_docs": state.get("graded_relevant_docs", 0),
        "total_hops": total_hops,
        "hop_queries": hop_queries,
    }

    return {
        "final_answer": state.get("generation", ""),
        "metadata": final_metadata
    }


# =============================
# 边函数(插桩版)
# =============================

@instrument_edge("should_continue_hop")
def should_continue_hop(state: Hop2RagState) -> str:
    """决定是继续下一跳还是生成最终答案"""
    current_hop = state.get("current_hop", 0)
    max_hops = state.get("max_hops", 5)

    if current_hop >= max_hops:
        return "finalize_answer"

    decision_key = f"hop_{current_hop - 1}_decision"
    decision = state.get("metadata", {}).get(decision_key, "no")

    if decision == "yes":
        return "finalize_answer"
    else:
        return "continue_hop"


# =============================
# 构建图
# =============================

def build_hop2_rag() -> StateGraph:
    """构建Hop2Rag工作流"""
    workflow = StateGraph(Hop2RagState)

    # 添加节点
    workflow.add_node("initialize", initialize_node)
    workflow.add_node("decompose", decompose_node)
    workflow.add_node("retrieve_hop", retrieve_hop_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("accumulate", accumulate_node)
    workflow.add_node("extract_clues", extract_clues_node)
    workflow.add_node("decide", decide_node)
    workflow.add_node("generate_final", generate_final_node)
    workflow.add_node("extract_supporting_facts_fast", extract_sp_fast_node)
    workflow.add_node("finalize", finalize_node)

    # 构建流程
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "decompose")

    # 跳数循环
    workflow.add_edge("decompose", "retrieve_hop")
    workflow.add_edge("retrieve_hop", "rerank")
    workflow.add_edge("rerank", "accumulate")
    workflow.add_edge("accumulate", "extract_clues")
    workflow.add_edge("extract_clues", "decide")

    # 条件路由
    workflow.add_conditional_edges(
        "decide",
        should_continue_hop,
        {
            "continue_hop": "decompose",
            "finalize_answer": "generate_final"
        }
    )

    # 最终生成流程
    workflow.add_edge("generate_final", "extract_supporting_facts_fast")
    workflow.add_edge("extract_supporting_facts_fast", "finalize")
    workflow.add_edge("finalize", END)

    return workflow


def get_hop2_rag_app():
    """获取编译后的Hop2Rag应用"""
    workflow = build_hop2_rag()
    return workflow.compile()


def save_workflow_graph(output_path: str):
    """保存工作流图到文件"""
    try:
        workflow = build_hop2_rag()
        app = workflow.compile()

        # 尝试使用 draw_mermaid_png
        try:
            png_data = app.get_graph().draw_mermaid_png()
            with open(output_path, "wb") as f:
                f.write(png_data)
            return True
        except Exception:
            pass

        # 尝试使用 draw_png
        try:
            png_data = app.get_graph().draw_png()
            with open(output_path, "wb") as f:
                f.write(png_data)
            return True
        except Exception:
            pass

        return False
    except Exception:
        return False


# =============================
# 便捷调用接口
# =============================

def run_hop2_rag(
    question: str,
    persist_dir: str,
    collection_name: str = "hotpot_fullwiki",
    k: int = 10,
    max_hops: int = 5
) -> Dict[str, Any]:
    """
    运行Hop2Rag查询

    Args:
        question: 用户问题
        persist_dir: Chroma持久化目录
        collection_name: 集合名称
        k: 每跳检索文档数
        max_hops: 最大跳数

    Returns:
        包含完整调试信息的字典:
        - answer: 最终答案
        - supporting_facts: 支撑事实
        - metadata: 元数据（含 aggregated_usage）
        - documents: 最终使用的文档
        - hop_evidence: 每跳提取的证据（截断）
        - intermediate_answers: 每跳的中间答案
        - hop_documents: 每跳检索的文档列表
        - timing_ms: 总耗时
        - aggregated_usage: 真实 token usage 汇总
    """
    app = get_hop2_rag_app()

    request_id = str(uuid.uuid4())

    inputs = {
        "question": question,
        "custom_retriever_config": {
            "persist_dir": persist_dir,
            "collection_name": collection_name,
            "k": str(k),
            "max_hops": str(max_hops)
        }
    }

    # 初始化 usage 聚合并设置 contextvar
    _usage_aggregator.init_request(request_id)
    token = _current_request_id.set(request_id)

    try:
        start = time.perf_counter()
        # 传入 callbacks 以收集 usage
        result = app.invoke(
            inputs,
            config={
                "recursion_limit": 100,
                "callbacks": get_usage_callbacks(),
            }
        )
        total_ms = (time.perf_counter() - start) * 1000

        # 获取聚合后的 usage
        aggregated_usage = _usage_aggregator.get_aggregated_usage(request_id)

        # 提取元数据
        metadata = result.get("metadata", {})
        total_hops = metadata.get("total_hops", 0)
        hop_queries = metadata.get("hop_queries", [])

        # 构造精简的 JSONL record（不包含大文本）
        record = {
            "request_id": request_id,
            "total_ms": round(total_ms, 2),
            "total_hops": total_hops,
            "hop_queries": [q[:100] for q in hop_queries],  # 截断查询
            "data_source": metadata.get("data_source", "unknown"),
            "used_web_search": metadata.get("used_web_search", False),
            "graded_relevant_docs": metadata.get("graded_relevant_docs", 0),
            "aggregated_usage": aggregated_usage,
            "node_ms": {nk: round(v, 2) for nk, v in result.get("_timings_ms", {}).items()},
            "edge_ms": {ek: round(v, 2) for ek, v in result.get("_edge_timings_ms", {}).items()},
            "config": {"k": k, "max_hops": max_hops},
        }
        _tracker.log_request(record)

        # 将 aggregated_usage 加入 metadata
        metadata["aggregated_usage"] = aggregated_usage

        # 处理 hop_evidence（截断以避免大文本）
        hop_evidence = result.get("hop_evidence", [])
        hop_evidence_truncated = [ev[:200] if isinstance(ev, str) else str(ev)[:200] for ev in hop_evidence]

        # 返回结果
        return {
            "answer": result.get("final_answer", ""),
            "supporting_facts": result.get("supporting_facts", []),
            "metadata": metadata,
            "documents": result.get("documents", []),
            "hop_evidence": hop_evidence_truncated,
            "intermediate_answers": result.get("intermediate_answers", []),
            "hop_documents": result.get("hop_documents", []),
            "hop_queries": hop_queries,
            "timing_ms": total_ms,
            "aggregated_usage": aggregated_usage,
        }
    finally:
        # 清理 contextvar 和 usage 数据
        _current_request_id.reset(token)
        _usage_aggregator.cleanup_request(request_id)


if __name__ == "__main__":
    import sys
    import os

    persist_dir = os.environ.get("AGRAG_PERSIST_DIR", "")
    collection_name = os.environ.get("AGRAG_COLLECTION_NAME", "hotpot_fullwiki")

    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
    else:
        q = "Who is the director of the movie that won Best Picture at the 2020 Oscars?"

    print(f"Question: {q}")
    result = run_hop2_rag(
        question=q,
        persist_dir=persist_dir,
        collection_name=collection_name,
        k=10,
        max_hops=3
    )
    print(f"Answer: {result['answer']}")
    print(f"Supporting Facts: {result['supporting_facts']}")
    print(f"Total Hops: {result['metadata'].get('total_hops', 'N/A')}")
    print(f"Total Time: {result['timing_ms']:.2f}ms")
