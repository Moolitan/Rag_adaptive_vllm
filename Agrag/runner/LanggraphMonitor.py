import time
from enum import Enum
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


class NodeType(str, Enum):
    """节点类型"""
    LLM = "llm"              # LLM 推理节点
    RETRIEVER = "retriever"  # 向量库检索节点
    CPU = "cpu"              # 非 LLM 的 CPU 计算节点


# LLM 相关的节点名称（会调用 LLM）
LLM_NODE_NAMES = {
    "decompose",
    "extract_clues",
    "decide",
    "generate_final",
    "generate",
}

# 检索相关的节点名称
RETRIEVER_NODE_NAMES = {
    "retrieve",
    "retrieve_hop",
    "search",
}


class DataCollector(BaseCallbackHandler):
    """
    LangGraph Node 级别的数据收集器。

    特点：
      - 只统计 LangGraph node 级别，不细分内部 chain
      - 每个 node 有类型属性：llm / retriever / cpu
      - 通过 langgraph_node 元数据识别真正的节点边界
      - 记录每个 LLM call 的 prompt 和 token 数量
    """

    def __init__(self, debug: bool = False, track_prompts: bool = True, encoding_name: str = "cl100k_base"):
        self.debug = debug
        self.track_prompts = track_prompts
        # run_id -> 节点执行信息（用 run_id 作为 key 而不是 node_name）
        self.active_runs: Dict[str, Dict[str, Any]] = {}
        # run_id -> 所属的 node run_id（用于追踪嵌套关系）
        self.run_to_parent: Dict[str, str] = {}
        # 最终记录
        self.records: List[Dict[str, Any]] = []
        # LLM call 记录（每个 LLM 调用的详细信息）
        self.llm_calls: List[Dict[str, Any]] = []

        # 初始化 tokenizer
        self.tokenizer = None
        if TIKTOKEN_AVAILABLE and track_prompts:
            try:
                self.tokenizer = tiktoken.get_encoding(encoding_name)
            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] Failed to load tiktoken encoding '{encoding_name}': {e}")
                self.tokenizer = None

    def _safe_str(self, x: Any) -> Optional[str]:
        if x is None:
            return None
        try:
            s = str(x)
            return s if s else None
        except Exception:
            return None

    def _count_tokens(self, text: str) -> int:
        """统计文本的 token 数量"""
        if not text:
            return 0
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        # 如果没有 tokenizer，使用简单估算（1 token ≈ 4 字符）
        return len(text) // 4

    def _get_langgraph_node(self, kwargs: Dict[str, Any]) -> Optional[str]:
        """从 metadata 或 tags 中提取 langgraph node 名称"""
        md = kwargs.get("metadata") or {}

        # 检查多种可能的键名
        candidate_keys = [
            "langgraph_node",
            "langgraph:node",
            "langgraph.node",
            "graph_node",
            "node",
            "node_name",
            "name",
        ]
        for key in candidate_keys:
            v = md.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()

        # 从 tags 中提取
        tags = kwargs.get("tags") or []
        for t in tags:
            if not isinstance(t, str):
                continue
            for prefix in ("langgraph_node:", "langgraph:node:", "node:", "graph:"):
                if t.startswith(prefix):
                    name = t.split(":", 1)[1].strip()
                    if name:
                        return name

        return None

    def _classify_node_type(self, node_name: str) -> NodeType:
        """根据节点名称判断节点类型"""
        name_lower = node_name.lower()

        for llm_name in LLM_NODE_NAMES:
            if llm_name in name_lower:
                return NodeType.LLM

        for ret_name in RETRIEVER_NODE_NAMES:
            if ret_name in name_lower:
                return NodeType.RETRIEVER

        return NodeType.CPU

    def _find_node_run_id(self, run_id: str) -> Optional[str]:
        """向上查找所属的节点 run_id"""
        # 直接是节点
        if run_id in self.active_runs:
            return run_id
        # 查找父节点
        parent_id = self.run_to_parent.get(run_id)
        if parent_id:
            return self._find_node_run_id(parent_id)
        return None

    # ---------------------------
    # Chain 回调
    # ---------------------------

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        run_id = self._safe_str(kwargs.get("run_id"))
        parent_run_id = self._safe_str(kwargs.get("parent_run_id"))
        if not run_id:
            return

        node_name = self._get_langgraph_node(kwargs)

        if node_name:
            # 这是一个 LangGraph node
            node_type = self._classify_node_type(node_name)
            self.active_runs[run_id] = {
                "start_time": time.time(),
                "node_name": node_name,
                "node_type": node_type,
                "has_llm_call": False,
                "has_retriever_call": False,
                "llm_latency": 0.0,
                "retriever_latency": 0.0,
                "doc_count": 0,
            }
            if self.debug:
                print(f"[DEBUG] START node: {node_name}, type: {node_type.value}, run_id: {run_id[:8]}...")
        else:
            # 嵌套的 chain，记录父子关系
            if parent_run_id:
                self.run_to_parent[run_id] = parent_run_id

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        run_id = self._safe_str(kwargs.get("run_id"))
        if not run_id:
            return

        # 检查是否是一个节点的结束
        if run_id in self.active_runs:
            info = self.active_runs.pop(run_id)
            end_time = time.time()
            latency = end_time - info["start_time"]

            # 根据实际执行情况更新节点类型
            actual_type = info["node_type"]
            if info["has_llm_call"]:
                actual_type = NodeType.LLM
            elif info["has_retriever_call"]:
                actual_type = NodeType.RETRIEVER

            record = {
                "event": "node_execution",
                "node_name": info["node_name"],
                "node_type": actual_type.value,
                "latency": latency,
                "timestamp": end_time,
            }

            if info["has_llm_call"]:
                record["llm_latency"] = info["llm_latency"]

            if info["has_retriever_call"]:
                record["retriever_latency"] = info["retriever_latency"]
                record["doc_count"] = info["doc_count"]

            self.records.append(record)

            if self.debug:
                print(f"[DEBUG] END node: {info['node_name']}, type: {actual_type.value}, latency: {latency*1000:.1f}ms")
        else:
            # 清理嵌套关系
            if run_id in self.run_to_parent:
                del self.run_to_parent[run_id]

    # ---------------------------
    # LLM 回调
    # ---------------------------

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        run_id = self._safe_str(kwargs.get("run_id"))
        parent_run_id = self._safe_str(kwargs.get("parent_run_id"))
        if not run_id:
            return

        # 记录父子关系
        if parent_run_id:
            self.run_to_parent[run_id] = parent_run_id

        # 找到所属的节点
        node_run_id = self._find_node_run_id(parent_run_id) if parent_run_id else None

        # 记录 LLM call 的详细信息
        if self.track_prompts and prompts:
            # 合并所有 prompts（通常只有一个）
            full_prompt = "\n".join(prompts)
            prompt_tokens = self._count_tokens(full_prompt)

            llm_call_info = {
                "run_id": run_id,
                "node_run_id": node_run_id,
                "node_name": self.active_runs[node_run_id]["node_name"] if node_run_id and node_run_id in self.active_runs else None,
                "prompt": full_prompt if len(full_prompt) < 10000 else full_prompt[:10000] + "...[truncated]",  # 限制长度
                "prompt_tokens": prompt_tokens,
                "timestamp": time.time(),
            }
            self.llm_calls.append(llm_call_info)

            if self.debug:
                print(f"[DEBUG] llm_call: node={llm_call_info['node_name']}, tokens={prompt_tokens}")

        if node_run_id and node_run_id in self.active_runs:
            self.active_runs[node_run_id]["llm_start_time"] = time.time()
            if self.debug and not self.track_prompts:
                print(f"[DEBUG] llm_start: node={self.active_runs[node_run_id]['node_name']}")
        elif self.debug and not self.track_prompts:
            print(f"[DEBUG] llm_start: parent_node=None (parent_run_id={parent_run_id[:8] if parent_run_id else None}...)")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        run_id = self._safe_str(kwargs.get("run_id"))
        parent_run_id = self._safe_str(kwargs.get("parent_run_id"))
        if not run_id:
            return

        # 找到所属的节点
        node_run_id = self._find_node_run_id(parent_run_id) if parent_run_id else None

        if node_run_id and node_run_id in self.active_runs:
            info = self.active_runs[node_run_id]
            info["has_llm_call"] = True

            if "llm_start_time" in info:
                llm_latency = time.time() - info["llm_start_time"]
                info["llm_latency"] += llm_latency
                del info["llm_start_time"]

        # 清理
        if run_id in self.run_to_parent:
            del self.run_to_parent[run_id]

    # ---------------------------
    # Retriever 回调
    # ---------------------------

    def on_retriever_start(self, serialized: Dict[str, Any], query: str, **kwargs: Any) -> None:
        run_id = self._safe_str(kwargs.get("run_id"))
        parent_run_id = self._safe_str(kwargs.get("parent_run_id"))
        if not run_id:
            return

        if parent_run_id:
            self.run_to_parent[run_id] = parent_run_id

        node_run_id = self._find_node_run_id(parent_run_id) if parent_run_id else None

        if node_run_id and node_run_id in self.active_runs:
            self.active_runs[node_run_id]["retriever_start_time"] = time.time()
            if self.debug:
                print(f"[DEBUG] retriever_start: node={self.active_runs[node_run_id]['node_name']}")
        elif self.debug:
            print(f"[DEBUG] retriever_start: parent_node=None")

    def on_retriever_end(self, documents: List[Any], **kwargs: Any) -> None:
        run_id = self._safe_str(kwargs.get("run_id"))
        parent_run_id = self._safe_str(kwargs.get("parent_run_id"))
        if not run_id:
            return

        node_run_id = self._find_node_run_id(parent_run_id) if parent_run_id else None

        if node_run_id and node_run_id in self.active_runs:
            info = self.active_runs[node_run_id]
            info["has_retriever_call"] = True
            info["doc_count"] = len(documents) if documents else 0

            if "retriever_start_time" in info:
                retriever_latency = time.time() - info["retriever_start_time"]
                info["retriever_latency"] += retriever_latency
                del info["retriever_start_time"]

        if run_id in self.run_to_parent:
            del self.run_to_parent[run_id]

    # ---------------------------
    # Tool 回调
    # ---------------------------

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        run_id = self._safe_str(kwargs.get("run_id"))
        parent_run_id = self._safe_str(kwargs.get("parent_run_id"))
        if run_id and parent_run_id:
            self.run_to_parent[run_id] = parent_run_id

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        run_id = self._safe_str(kwargs.get("run_id"))
        if run_id and run_id in self.run_to_parent:
            del self.run_to_parent[run_id]

    # ---------------------------
    # 获取结果
    # ---------------------------

    def get_records(self) -> List[Dict[str, Any]]:
        """获取所有记录"""
        return self.records.copy()

    def get_llm_calls(self) -> List[Dict[str, Any]]:
        """获取所有 LLM call 记录"""
        return self.llm_calls.copy()

    def get_prompt_token_distribution(self) -> Dict[str, Any]:
        """获取 prompt token 分布统计"""
        if not self.llm_calls:
            return {
                "total_calls": 0,
                "total_tokens": 0,
                "min_tokens": 0,
                "max_tokens": 0,
                "mean_tokens": 0,
                "median_tokens": 0,
                "token_counts": [],
            }

        token_counts = [call["prompt_tokens"] for call in self.llm_calls]
        token_counts_sorted = sorted(token_counts)

        return {
            "total_calls": len(self.llm_calls),
            "total_tokens": sum(token_counts),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "mean_tokens": sum(token_counts) / len(token_counts),
            "median_tokens": token_counts_sorted[len(token_counts_sorted) // 2],
            "token_counts": token_counts,
        }

    def get_summary(self) -> Dict[str, Any]:
        """获取统计摘要"""
        total_latency = sum(r["latency"] for r in self.records)
        llm_latency = sum(r.get("llm_latency", 0) for r in self.records)
        retriever_latency = sum(r.get("retriever_latency", 0) for r in self.records)

        llm_nodes = [r for r in self.records if r["node_type"] == NodeType.LLM.value]
        retriever_nodes = [r for r in self.records if r["node_type"] == NodeType.RETRIEVER.value]
        cpu_nodes = [r for r in self.records if r["node_type"] == NodeType.CPU.value]

        return {
            "total_nodes": len(self.records),
            "total_latency_sec": total_latency,
            "llm_nodes": len(llm_nodes),
            "llm_latency_sec": llm_latency,
            "retriever_nodes": len(retriever_nodes),
            "retriever_latency_sec": retriever_latency,
            "cpu_nodes": len(cpu_nodes),
            "cpu_latency_sec": total_latency - llm_latency - retriever_latency,
        }

    def clear(self) -> None:
        """清空所有记录"""
        self.active_runs.clear()
        self.run_to_parent.clear()
        self.records.clear()
        self.llm_calls.clear()
