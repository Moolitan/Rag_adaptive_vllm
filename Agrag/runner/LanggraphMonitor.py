import time
from typing import Any, Dict, List
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

class DataCollector(BaseCallbackHandler):
    def __init__(self):
        self.starts = {}
        # 这里定义一个容器来存数据
        self.records = [] 

    # --- 监控 LLM 开始 ---
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        run_id = kwargs.get("run_id")
        name = serialized.get("name") if serialized else None
        self.starts[run_id] = {
            "start_time": time.time(),
            "type": "llm",
            "name": name,
            "prompt_preview": prompts[0][:50] if prompts else ""
        }

    # --- 监控 LLM 结束 ---
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        run_id = kwargs.get("run_id")
        start_data = self.starts.pop(run_id, {})
        end_time = time.time()
        
        # 提取生成内容
        generation = response.generations[0][0].text
        # 尝试提取 token usage (取决于 vLLM 是否返回)
        usage = response.llm_output.get("token_usage", {}) if response.llm_output else {}

        # 把整理好的数据存进 self.records
        record = {
            "event": "llm_call",
            "model": start_data.get("name"),
            "latency": end_time - start_data.get("start_time", end_time),
            "timestamp": end_time,
            "prompt": start_data.get("prompt_preview"),
            "response": generation, # 这里拿到了 LLM 的回答
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0)
        }
        self.records.append(record)

    # --- 监控 Tool/Node 开始 ---
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        run_id = kwargs.get("run_id")
        # serialized 可能为 None
        if serialized is None:
            return
        name = serialized.get("name")
        # 过滤掉 LangGraph 内部的一些杂项 chain，只关注主要的 Node
        if name and "LangGraph" not in name:
            self.starts[run_id] = {
                "start_time": time.time(),
                "type": "node",
                "name": name
            }

    # --- 监控 Tool/Node 结束 ---
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        run_id = kwargs.get("run_id")
        start_data = self.starts.pop(run_id, None)
        if start_data: # 只有在 start 里记录过的才处理
            end_time = time.time()
            record = {
                "event": "node_execution",
                "node_name": start_data["name"],
                "latency": end_time - start_data["start_time"],
                "timestamp": end_time,
                # "outputs": outputs # 如果需要中间数据，可以取消注释
            }
            self.records.append(record)


