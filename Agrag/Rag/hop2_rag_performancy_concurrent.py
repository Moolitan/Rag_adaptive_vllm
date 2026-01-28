"""
Hop2Rag - Concurrent Version

支持传入独立的 DataCollector，用于并发测试场景
保留原有的 hop2_rag_performancy.py 不变
"""

import re
import math
from typing import TypedDict, List, Dict, Any, Tuple, Optional
from collections import Counter

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.graph import StateGraph, START, END

from runner.LanggraphMonitor import DataCollector

# =============================
# 状态定义
# =============================

class Hop2RagState(TypedDict, total=False):
    """Hop2Rag状态定义"""
    # 核心输入输出
    question: str
    generation: str
    documents: List[Document]

    # 多跳控制
    current_hop: int
    max_hops: int
    hop_k: int
    decision: str

    # 跳数历史追踪
    hop_queries: List[str]
    hop_documents: List[List[Document]]
    hop_evidence: List[str]

    # 中间推理
    all_graded_documents: List[Document]

    # Supporting facts
    supporting_facts: List

    # 最终输出
    final_answer: str

    # 自定义检索配置
    custom_retriever_config: Dict[str, str]


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
# 提示词定义（与原版相同）
# =============================

QUESTION_DECOMPOSER_PROMPT = """You are a question decomposition expert for multi-hop reasoning tasks.

Current hop: {current_hop}
Original question: {original_question}
Evidence so far:
{evidence}

Previous queries:
{previous_queries}

Instructions:
- Generate a NEW search query for the NEXT retrieval hop
- The query MUST be different from all previous queries
- Hop 0 (initial): extract the first key entity or concept to search for
- Hop 1+: based on collected evidence, determine what entity or attribute to search next
- For bridge questions: identify the intermediate entity, then search toward the final answer
- For comparison questions: if hop 0 searched entity A, hop 1 should search entity B or a comparable attribute
- If you are uncertain what to search next, return the original question as the query (fallback)

Return JSON:
{{
  "query": "search query for this hop",
  "reasoning": "why this query is useful for the next hop"
}}
"""

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

HOP_DECISION_PROMPT = """Decide whether to continue retrieval for another hop, or stop and answer now.

Original question:
{question}

Current hop (0-indexed): {current_hop}
Max hops: {max_hops}

Evidence collected so far:
{evidence}


Instructions:
- Output "stop" if the current evidence is sufficient to answer the original question in a reasonable and coherent way,
  even if some minor details might be missing.
- Output "continue" only if a clearly important entity, attribute, or comparison target is missing or unclear,
  and another retrieval is likely to meaningfully improve the answer.
- If the evidence already forms a plausible answer path, prefer "stop" over "continue".
- Avoid continuing solely for completeness or extra confirmation.

Return JSON:
{{"decision": "continue" or "stop", "reasoning": "brief justification"}}
"""

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
# BM25 Reranker
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


# =============================
# Sentence Selector
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


# =============================
# 向量库配置（线程安全）
# =============================
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from threading import Lock

_custom_retrievers = {}
_retriever_lock = Lock()


def get_custom_retriever(persist_dir: str, collection_name: str, k: int = 10):
    """获取自定义retriever（线程安全）"""
    cache_key = f"{persist_dir}:{collection_name}:{k}"

    # 使用锁保护缓存访问
    with _retriever_lock:
        if cache_key not in _custom_retrievers:
            if not persist_dir:
                raise ValueError("persist_dir is required")
            if not collection_name:
                raise ValueError("collection_name is required")

            embedding = HuggingFaceEmbeddings(
                model_name="/mnt/Large_Language_Model_Lab_1/模型/rag_models/BAAI-bge-base-en-v1.5",
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
# 链构建（线程安全 - 使用缓存）
# =============================

_question_decomposer = None
_clue_extractor = None
_hop_decision_chain = None
_multi_hop_rag_chain = None
_chain_lock = Lock()


def get_question_decomposer():
    """获取问题分解链（线程安全）"""
    global _question_decomposer
    with _chain_lock:
        if _question_decomposer is None:
            prompt = PromptTemplate(
                template=QUESTION_DECOMPOSER_PROMPT,
                input_variables=["current_hop", "original_question", "evidence", "previous_queries"]
            )
            _question_decomposer = prompt | get_llm(json_mode=True) | JsonOutputParser()
        return _question_decomposer


def get_clue_extractor():
    """获取线索提取链（线程安全）"""
    global _clue_extractor
    with _chain_lock:
        if _clue_extractor is None:
            prompt = PromptTemplate(
                template=CLUE_EXTRACTOR_PROMPT,
                input_variables=["question", "current_hop", "documents"]
            )
            _clue_extractor = prompt | get_llm(json_mode=True) | JsonOutputParser()
        return _clue_extractor


def get_hop_decision_chain():
    """获取跳数决策链（线程安全）"""
    global _hop_decision_chain
    with _chain_lock:
        if _hop_decision_chain is None:
            prompt = PromptTemplate(
                template=HOP_DECISION_PROMPT,
                input_variables=["question", "current_hop", "max_hops", "evidence"]
            )
            _hop_decision_chain = prompt | get_llm(json_mode=True) | JsonOutputParser()
        return _hop_decision_chain


def get_multi_hop_rag_chain():
    """获取多跳RAG生成链（线程安全）"""
    global _multi_hop_rag_chain
    with _chain_lock:
        if _multi_hop_rag_chain is None:
            prompt = PromptTemplate(
                template=MULTI_HOP_RAG_PROMPT,
                input_variables=["question", "evidence", "context"]
            )
            _multi_hop_rag_chain = prompt | get_llm() | StrOutputParser()
        return _multi_hop_rag_chain


# =============================
# 节点函数（与原版相同）
# =============================

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
        "all_graded_documents": [],
    }


def decompose_node(state: Hop2RagState) -> Dict[str, Any]:
    current_hop = state.get("current_hop", 0)
    original_question = state["question"]
    evidence = state.get("hop_evidence", [])
    hop_queries = state.get("hop_queries", [])

    evidence_text = "\n".join([f"Hop {i}: {ev}" for i, ev in enumerate(evidence)]) if evidence else "None"
    previous_queries_text = ", ".join([f'"{q}"' for q in hop_queries]) if hop_queries else "None"

    decomposer = get_question_decomposer()
    result = decomposer.invoke({
        "current_hop": current_hop,
        "original_question": original_question,
        "evidence": evidence_text,
        "previous_queries": previous_queries_text
    })

    raw_query = result.get("query", "")
    next_query = raw_query.strip() if isinstance(raw_query, str) else ""

    if not next_query:
        next_query = original_question.strip()

    hop_queries_updated = hop_queries + [next_query]

    return {
        "hop_queries": hop_queries_updated
    }


def retrieve_hop_node(state: Hop2RagState) -> Dict[str, Any]:
    """检索当前跳的文档"""
    current_hop = state.get("current_hop", 0)
    hop_queries = state.get("hop_queries", [])
    hop_k = state.get("hop_k", 10)

    query = hop_queries[current_hop]
    cfg = state.get("custom_retriever_config", {})
    custom_retriever = get_custom_retriever(
        persist_dir=cfg.get("persist_dir", ""),
        collection_name=cfg.get("collection_name", "hotpotqa_fullwiki"),
        k=hop_k
    )
    docs = custom_retriever.invoke(query)
    return {
        "documents": docs,
    }


def rerank_node(state: Hop2RagState) -> Dict[str, Any]:
    """使用BM25重排序文档"""
    docs = state.get("documents", [])
    if not docs:
        return {"documents": []}

    current_hop = state.get("current_hop", 0)
    hop_queries = state.get("hop_queries", [])
    original_question = state["question"]

    query = hop_queries[current_hop]
    rerank_query = f"{original_question} || {query}"

    reranker = BM25Reranker()
    scored_docs = reranker.score_documents(rerank_query, docs)

    top_k = min(5, len(scored_docs))
    kept_docs = [doc for doc, score in scored_docs[:top_k]]

    return {
        "documents": kept_docs
    }


def accumulate_node(state: Hop2RagState) -> Dict[str, Any]:
    """累积已评分的相关文档"""
    current_docs = state.get("documents", [])
    all_graded = state.get("all_graded_documents", [])
    hop_documents = state.get("hop_documents", [])

    existing_contents = {d.page_content for d in all_graded}
    new_docs = [d for d in current_docs[:3] if d.page_content not in existing_contents]
    all_graded_updated = all_graded + new_docs

    hop_documents_updated = hop_documents + [current_docs]

    return {
        "all_graded_documents": all_graded_updated,
        "hop_documents": hop_documents_updated
    }


def extract_clues_node(state: Hop2RagState) -> Dict[str, Any]:
    """线索提取节点"""
    current_hop = state.get("current_hop", 0)
    question = state["question"]
    documents = state.get("documents", [])
    hop_evidence = state.get("hop_evidence", [])

    docs_text = "\n\n".join([
        f"Doc {i+1}: {d.page_content[:300]}"
        for i, d in enumerate(documents[:10])
    ])

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

    return {
        "hop_evidence": hop_evidence_updated,
    }


def decide_node(state: Hop2RagState) -> Dict[str, Any]:
    """跳数决策节点"""
    current_hop = state.get("current_hop", 0)
    max_hops = state.get("max_hops", 5)
    question = state["question"]
    hop_evidence = state.get("hop_evidence", [])

    evidence_text = "\n".join([f"Hop {i}: {ev}" for i, ev in enumerate(hop_evidence)]) or "None"

    decision_chain = get_hop_decision_chain()
    result = decision_chain.invoke({
        "question": question,
        "current_hop": current_hop,
        "max_hops": max_hops,
        "evidence": evidence_text
    })

    raw = result.get("decision", "continue")
    if isinstance(raw, str):
        raw = raw.strip().lower()
    else:
        raw = "continue"

    if raw not in ("continue", "stop"):
        raw = "continue"

    decision = "stop" if current_hop >= max_hops - 1 else raw
    next_hop = current_hop + 1 if decision == "continue" else current_hop

    return {
        "current_hop": next_hop,
        "decision": decision,
    }


def generate_final_node(state: Hop2RagState) -> Dict[str, Any]:
    """多跳最终生成节点"""
    question = state["question"]
    hop_evidence = state.get("hop_evidence", [])
    hop_queries = state.get("hop_queries", [])
    all_graded = state.get("all_graded_documents", [])
    hop_documents = state.get("hop_documents", [])

    evidence_parts = []
    for i, ev in enumerate(hop_evidence):
        query = hop_queries[i] if i < len(hop_queries) else ""
        evidence_parts.append(f"Hop {i} (Query: {query}):\n{ev}")
    evidence_text = "\n\n".join(evidence_parts)

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


def extract_sp_fast_node(state: Hop2RagState) -> Dict[str, Any]:
    """使用BM25快速提取supporting facts"""
    question = state["question"]
    answer = state.get("generation", "")
    documents = state.get("documents", [])

    if not documents:
        return {"supporting_facts": []}

    selector = SentenceSelector(top_m=3)
    sp_facts = selector.extract_supporting_facts(question, answer, documents)

    supporting_facts = [[title, idx] for title, idx in sp_facts]

    return {"supporting_facts": supporting_facts}


def finalize_node(state: Hop2RagState) -> Dict[str, Any]:
    """最终化输出"""
    return {
        "final_answer": state.get("generation", ""),
    }


def should_continue_hop(state: Hop2RagState) -> str:
    """路由"""
    return "continue_hop" if state.get("decision", "continue") == "continue" else "finalize_answer"


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


# =============================
# 便捷调用接口（并发版本）
# =============================

def run_hop2_rag_with_collector(
    question: str,
    persist_dir: str,
    collection_name: str = "hotpot_fullwiki",
    k: int = 10,
    max_hops: int = 5,
    collector: DataCollector = None
) -> Dict[str, Any]:
    """
    运行 Hop2Rag（支持传入独立的 DataCollector）

    Args:
        question: 用户问题
        persist_dir: 向量库路径
        collection_name: 集合名称
        k: 检索 K 值
        max_hops: 最大跳数
        collector: DataCollector 实例（如果为 None，创建临时的）

    Returns:
        包含 answer 和 current_hop 的字典
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

    # 如果没有提供 collector，创建临时的
    if collector is None:
        collector = DataCollector(track_prompts=False)

    result = app.invoke(
        inputs,
        config={"recursion_limit": 100, "callbacks": [collector]}
    )

    return {
        "answer": result.get("final_answer", ""),
        "current_hop": result.get("current_hop", 0),
    }


if __name__ == "__main__":
    import sys
    import os

    persist_dir = os.environ.get("AGRAG_PERSIST_DIR", "")
    collection_name = os.environ.get("AGRAG_COLLECTION_NAME", "hotpotqa_fullwiki")

    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
    else:
        q = "Who is the director of the movie that won Best Picture at the 2020 Oscars?"

    print(f"Question: {q}")

    # 创建独立的 collector
    collector = DataCollector(track_prompts=True, encoding_name="cl100k_base")

    result = run_hop2_rag_with_collector(
        question=q,
        persist_dir=persist_dir,
        collection_name=collection_name,
        k=10,
        max_hops=3,
        collector=collector
    )

    print(f"Answer: {result['answer']}")
    print(f"Current hop: {result['current_hop']}")

    # 查看 LLM calls
    llm_calls = collector.get_llm_calls()
    print(f"\nLLM calls: {len(llm_calls)}")
    for i, call in enumerate(llm_calls[:3], 1):
        print(f"  {i}. Node: {call['node_name']}, Tokens: {call['prompt_tokens']}")
