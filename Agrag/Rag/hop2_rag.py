import time
import json
import uuid
import re
from typing import TypedDict, List, Dict, Any, Tuple
from pathlib import Path
from functools import wraps

import numpy as np
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

    def enable(self, output_dir: str, filename: str = "hop2rag_timings.jsonl"):
        """启用性能追踪并设置输出目录"""
        self.close()  # 关闭之前的句柄
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.filename = filename
        self.jsonl_path = self.output_dir / self.filename
        self.file_handle = open(self.jsonl_path, "a", buffering=1<<20)
        self._enabled = True

    def disable(self):
        """禁用性能追踪"""
        self.close()
        self._enabled = False

    def log_request(self, record: Dict[str, Any]):
        """追加写入一条 JSONL 记录"""
        if self._enabled and self.file_handle:
            self.file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.file_handle.flush()  # 确保立即写入

    def clear_log(self):
        """清空日志文件"""
        if self.jsonl_path and self.jsonl_path.exists():
            self.close()
            self.jsonl_path.unlink()
            # 重新打开
            if self._enabled:
                self.file_handle = open(self.jsonl_path, "a", buffering=1<<20)

    def close(self):
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None

    def __del__(self):
        self.close()


# 全局追踪器(默认禁用，需要显式启用)
_tracker = PerformanceTracker()


def enable_instrumentation(output_dir: str, filename: str = "hop2rag_timings.jsonl"):
    """启用性能追踪

    Args:
        output_dir: 输出目录路径
        filename: 输出文件名
    """
    _tracker.enable(output_dir, filename)


def disable_instrumentation():
    """禁用性能追踪"""
    _tracker.disable()


def clear_instrumentation_log():
    """清空性能日志"""
    _tracker.clear_log()


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

# 问题分解提示词
QUESTION_DECOMPOSER_PROMPT = """You are a question decomposition expert for multi-hop reasoning tasks.

Current hop: {current_hop}
Original question: {original_question}
Evidence so far: {evidence}

Instructions:
- Hop 0 (initial): Extract the first entity/concept to search for
- Hop 1+: Based on evidence, determine what to search next
- For bridge questions: Find intermediate entity, then search for final answer
- For comparison questions: Identify each entity to compare

Return JSON: {{"query": "search query for this hop", "reasoning": "why this query"}}"""

# 线索提取提示词
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

# 跳数决策提示词
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

# 多跳RAG生成提示词
MULTI_HOP_RAG_PROMPT = """Answer a multi-hop question using evidence from multiple retrieval steps.

Original question: {question}

=== Evidence from multiple hops ===
{context}

Hop history:
{hop_history}

Instructions:
1. Synthesize information across ALL hops
2. For comparison questions: Compare attributes found in different hops
3. For bridge questions: Connect intermediate entity to final answer
4. Ground your answer in the evidence provided
5. Give a direct, concise answer

Answer:"""


# =============================
# Reranker实现(非LLM)
# =============================

class SimpleTFIDFEmbedder:
    """简单的TF-IDF嵌入器"""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(max_features=384, stop_words='english')
        self._fitted = False

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            self.vectorizer.fit(texts)
            self._fitted = True

        try:
            vectors = self.vectorizer.transform(texts).toarray()
        except:
            self.vectorizer.fit(texts)
            vectors = self.vectorizer.transform(texts).toarray()

        return vectors


class EmbeddingReranker:
    """基于embedding相似度的reranker"""

    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
        self.embedding_model = SimpleTFIDFEmbedder()

    def score_documents(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """批量评分文档"""
        if not documents:
            return []

        query_emb = self.embedding_model.encode([query])[0]
        # 把每篇 doc 截断前 500 字符
        doc_texts = [d.page_content[:500] for d in documents]
        # TF-IDF 编码为 doc 向量
        doc_embs = self.embedding_model.encode(doc_texts)

        scores = []
        for i, doc in enumerate(documents):
            similarity = self._cosine_similarity(query_emb, doc_embs[i])
            scores.append((doc, similarity))

        return scores

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


_reranker = None


def get_reranker(threshold: float = 0.3) -> EmbeddingReranker:
    """获取reranker实例"""
    global _reranker
    if _reranker is None:
        _reranker = EmbeddingReranker(threshold=threshold)
    return _reranker


# =============================
# Sentence Selector实现(非LLM)
# =============================

class SentenceSelector:
    """基于相似度的句子级supporting facts选择器"""

    def __init__(self, top_m: int = 3):
        self.top_m = top_m
        self.embedding_model = SimpleTFIDFEmbedder()

    def extract_supporting_facts(
        self,
        question: str,
        answer: str,
        documents: List[Document]
    ) -> List[Tuple[str, int]]:
        """提取supporting facts"""
        query = f"{question} {answer}"
        query_emb = self.embedding_model.encode([query])[0]

        all_sentences = []
        for doc_idx, doc in enumerate(documents):
            # title 兜底逻辑：优先从多个字段获取，仍无则用稳定占位符
            doc_title = (
                doc.metadata.get("title")
                or doc.metadata.get("page_title")
                or doc.metadata.get("wiki_title")
                or doc.metadata.get("source")
                or doc.metadata.get("doc_id")
                or doc.metadata.get("id")
            )
            if not doc_title or (isinstance(doc_title, str) and not doc_title.strip()):
                # 使用 page_content 的 hash 前8位作为稳定占位符
                content_hash = hex(abs(hash(doc.page_content)))[2:10]
                doc_title = f"Doc-{content_hash}"

            sentences = self._split_sentences(doc.page_content)

            for sent_idx, sent_text in enumerate(sentences):
                if len(sent_text.strip()) > 10:
                    all_sentences.append((doc_title, sent_idx, sent_text, doc))

        if not all_sentences:
            return []

        sent_texts = [s[2] for s in all_sentences]
        sent_embs = self.embedding_model.encode(sent_texts)

        scored = []
        for i, (doc_title, sent_idx, sent_text, doc_obj) in enumerate(all_sentences):
            similarity = self._cosine_similarity(query_emb, sent_embs[i])
            scored.append((doc_title, sent_idx, similarity, sent_text))

        scored.sort(key=lambda x: x[2], reverse=True)

        result = [(title, idx) for title, idx, score, text in scored[:self.top_m]]
        return result

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _cosine_similarity(a, b) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


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
            input_variables=["current_hop", "original_question", "evidence"]
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
            input_variables=["question", "context", "hop_history"]
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
    # 之前每一跳抽取出的中间证据
    evidence = state.get("hop_evidence", [])
    hop_queries = state.get("hop_queries", [])
    metadata = state.get("metadata", {})

    # 把"历史证据"整理成 LLM 可读文本，桥接真正发生的地方
    # Hop 0: Parasite - South Korean black comedy thriller film
    # Hop 1: Parasite won Best Picture at the 2020 Oscars
    evidence_text = "\n".join([f"Hop {i}: {ev}" for i, ev in enumerate(evidence)]) if evidence else "None"

    decomposer = get_question_decomposer()
    result = decomposer.invoke({
        "current_hop": current_hop,
        "original_question": original_question,
        "evidence": evidence_text
    })

    # 强约束: 对 query 做 strip，为空时 fallback 到 original_question
    raw_query = result.get("query", "")
    next_query = raw_query.strip() if isinstance(raw_query, str) else ""

    metadata_updated = metadata.copy()
    if not next_query:
        next_query = original_question.strip()
        metadata_updated[f"hop_{current_hop}_decompose_fallback"] = True
        metadata_updated[f"hop_{current_hop}_decompose_raw"] = result

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

    # 把本 hop 使用的 query 写入 metadata
    metadata_updated = metadata.copy()
    metadata_updated[f"hop_{current_hop}_retrieve_query"] = query

    return {
        "documents": docs,
        "data_source": "custom_vectorstore",
        "metadata": metadata_updated
    }


@instrument_node("grade_documents_reranker")
def grade_reranker_node(state: Hop2RagState) -> Dict[str, Any]:
    """使用reranker批量评分文档"""
    docs = state.get("documents", [])
    if not docs:
        return {"document_scores": []}

    # 使用本 hop 的 hop_query 作为 rerank query，而非 state["question"]
    current_hop = state.get("current_hop", 0)
    hop_queries = state.get("hop_queries", [])
    original_question = state["question"]
    metadata = state.get("metadata", {})

    # 获取本 hop 的 query；如果不存在则 fallback 到 original_question
    if current_hop < len(hop_queries):
        hop_query = hop_queries[current_hop]
    else:
        hop_query = original_question

    # 构建 rerank_query：混合 original_question + hop_query
    if hop_query != original_question:
        rerank_query = f"{original_question} || {hop_query}"
    else:
        rerank_query = original_question

    reranker = get_reranker(threshold=0.3)
    scored_docs = reranker.score_documents(rerank_query, docs)

    # 把 raw_score 转成 yes/no，并生成 document_scores
    document_scores = []
    for doc, score in scored_docs:
        binary_score = "yes" if score >= reranker.threshold else "no"
        document_scores.append({
            "document": doc,
            "score": binary_score,
            "raw_score": score
        })

    # 把 rerank_query 写入 metadata
    metadata_updated = metadata.copy()
    metadata_updated[f"hop_{current_hop}_rerank_query"] = rerank_query

    return {"document_scores": document_scores, "metadata": metadata_updated}


@instrument_node("filter_relevant_documents_reranker")
def filter_reranker_node(state: Hop2RagState) -> Dict[str, Any]:
    """根据reranker评分结果过滤文档"""
    # 取评分列表
    # 每个元素：{"document": Document, "score": "yes"/"no", "raw_score": float}
    scores = state.get("document_scores", [])
    # 过滤：只保留 score == "yes" 的 Document
    kept = [s["document"] for s in scores if s["score"] == "yes"]
    # 只要过滤掉任何一个（存在 no），就认为有不相关文档。
    has_irrelevant = len(kept) < len(scores)

    return {
        "documents": kept,
        "graded_relevant_docs": len(kept),
        "has_irrelevant_docs": has_irrelevant
    }


@instrument_node("accumulate_graded_documents")
def accumulate_node(state: Hop2RagState) -> Dict[str, Any]:
    """
        累积已评分的相关文档
        把本 hop 过滤后的相关文档加入全局累计集合 all_graded_documents,
        同时把本 hop 的文档列表追加到 hop_documents 里用于追踪。
    """
    current_hop = state.get("current_hop", 0)
    current_docs = state.get("documents", []) # 本 hop 的过滤后文档（来自 filter_reranker_node）
    all_graded = state.get("all_graded_documents", []) # 之前 hop 累积的文档池
    hop_documents = state.get("hop_documents", [])

    # 去重：用 page_content 作为唯一键
    # 避免跨 hop 重复把同一个 chunk 加进 all_graded_documents
    existing_contents = {d.page_content for d in all_graded}
    new_docs = [d for d in current_docs if d.page_content not in existing_contents]
    all_graded_updated = all_graded + new_docs # 只追加去重后的 new_docs

    # 不去重，原样记录本 hop 的文档列表
    # 用于后面分析每 hop 检索效果
    # 画论文图/做 ablation（每跳召回文档变化）
    hop_documents_updated = hop_documents + [current_docs]

    return {
        "all_graded_documents": all_graded_updated,
        "hop_documents": hop_documents_updated
    }


@instrument_node("extract_clues")
def extract_clues_node(state: Hop2RagState) -> Dict[str, Any]:
    """线索提取节点(保留LLM:推理核心)"""
    current_hop = state.get("current_hop", 0)
    question = state["question"]
    documents = state.get("documents", [])
    hop_evidence = state.get("hop_evidence", [])
    intermediate_answers = state.get("intermediate_answers", [])

    docs_text = "\n\n".join([
        f"Doc {i+1}: {d.page_content[:300]}" # 每篇只取前 300 字符
        for i, d in enumerate(documents[:10]) # 最多取 10 篇
    ])

    # 桥接的起点
    # 桥接实体被“消费” —— decompose_node 已经用掉了
    extractor = get_clue_extractor()
    result = extractor.invoke({
        "question": question,
        "current_hop": current_hop,
        "documents": docs_text
    })

    clues = result.get("clues", []) # 实体/线索抽取器
    summary = result.get("summary", "") # 本 hop 局部摘要器
    # 把 LLM 输出变成 hop_evidence 的一条记录
    evidence_text = f"{', '.join(clues)} - {summary}" if clues else summary

    hop_evidence_updated = hop_evidence + [evidence_text]
    intermediate_answers_updated = intermediate_answers + [summary]

    return {
        "hop_evidence": hop_evidence_updated,
        "intermediate_answers": intermediate_answers_updated
    }


@instrument_node("decide_next_hop")
def decide_node(state: Hop2RagState) -> Dict[str, Any]:
    """跳数决策节点(保留LLM:Agentic核心)"""
    current_hop = state.get("current_hop", 0)
    max_hops = state.get("max_hops", 2)
    question = state["question"]
    hop_evidence = state.get("hop_evidence", [])
    all_docs = state.get("all_graded_documents", [])
    metadata = state.get("metadata", {})

    evidence_text = "\n".join([f"Hop {i}: {ev}" for i, ev in enumerate(hop_evidence)])

    decision_chain = get_hop_decision_chain()
    result = decision_chain.invoke({
        "question": question,
        "current_hop": current_hop,
        "max_hops": max_hops,
        "evidence": evidence_text,
        "doc_count": len(all_docs)
    })

    decision = result.get("decision", "no")
    reasoning = result.get("reasoning", "")

    metadata_updated = metadata.copy()
    metadata_updated[f"hop_{current_hop}_decision"] = decision
    metadata_updated[f"hop_{current_hop}_reasoning"] = reasoning

    next_hop = current_hop + 1

    return {
        "current_hop": next_hop,
        "metadata": metadata_updated
    }


@instrument_node("generate_multi_hop_final")
def generate_final_node(state: Hop2RagState) -> Dict[str, Any]:
    """多跳最终生成节点(保留LLM:生成核心)"""
    question = state["question"]
    all_docs = state.get("all_graded_documents", [])
    hop_queries = state.get("hop_queries", [])
    hop_evidence = state.get("hop_evidence", [])

    context = "\n\n".join([
        f"[Doc {i+1}] {d.page_content}"
        for i, d in enumerate(all_docs)
    ])

    hop_history = []
    for i in range(len(hop_queries)):
        query = hop_queries[i] if i < len(hop_queries) else ""
        evidence = hop_evidence[i] if i < len(hop_evidence) else ""
        hop_history.append(f"Hop {i}: Query='{query}', Evidence='{evidence}'")
    hop_history_text = "\n".join(hop_history)

    rag_chain = get_multi_hop_rag_chain()
    result = rag_chain.invoke({
        "question": question,
        "context": context,
        "hop_history": hop_history_text
    })

    return {
        "generation": result,
        "documents": all_docs
    }


@instrument_node("extract_supporting_facts_fast")
def extract_sp_fast_node(state: Hop2RagState) -> Dict[str, Any]:
    """使用sentence selector快速提取supporting facts"""
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
    return {
        "final_answer": state.get("generation", ""),
        "metadata": {
            "data_source": state.get("data_source"),
            "used_web_search": state.get("used_web_search", False),
            "graded_relevant_docs": state.get("graded_relevant_docs", 0),
            "total_hops": state.get("current_hop", 0),
            "hop_queries": state.get("hop_queries", []),
        }
    }


# =============================
# 边函数(插桩版)
# =============================

@instrument_edge("should_continue_hop")
def should_continue_hop(state: Hop2RagState) -> str:
    """决定是继续下一跳还是生成最终答案"""
    current_hop = state.get("current_hop", 0)
    max_hops = state.get("max_hops", 5)

    # 硬性限制检查
    if current_hop >= max_hops:
        return "finalize_answer"

    # 文档检查
    all_docs = state.get("all_graded_documents", [])
    if len(all_docs) == 0 and current_hop > 0:
        return "finalize_answer"

    # LLM决策检查
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
    workflow.add_node("grade_reranker", grade_reranker_node)
    workflow.add_node("filter_docs", filter_reranker_node)
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
    workflow.add_edge("retrieve_hop", "grade_reranker")
    workflow.add_edge("grade_reranker", "filter_docs")
    workflow.add_edge("filter_docs", "accumulate")
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
        包含 final_answer, supporting_facts 和 metadata 的字典
    """
    app = get_hop2_rag_app()

    inputs = {
        "question": question,
        "custom_retriever_config": {
            "persist_dir": persist_dir,
            "collection_name": collection_name,
            "k": str(k),
            "max_hops": str(max_hops)
        }
    }

    # 测量总耗时
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    result = app.invoke(inputs)
    total_ms = (time.perf_counter() - start) * 1000

    # 记录性能日志
    record = {
        "request_id": request_id,
        "total_ms": round(total_ms, 2),
        "node_ms": {k: round(v, 2) for k, v in result.get("_timings_ms", {}).items()},
        "edge_ms": {k: round(v, 2) for k, v in result.get("_edge_timings_ms", {}).items()},
        "meta": {"k": k, "max_hops": max_hops, "question": question[:100]}
    }
    _tracker.log_request(record)

    return {
        "answer": result.get("final_answer", ""),
        "supporting_facts": result.get("supporting_facts", []),
        "metadata": result.get("metadata", {}),
        "documents": result.get("documents", []),
        "timing_ms": total_ms
    }


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
    print(f"Total Time: {result['timing_ms']:.2f}ms")
