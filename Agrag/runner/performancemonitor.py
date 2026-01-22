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

# ==============================================================================
import requests
import time

VLLM_METRICS_URL = "http://localhost:8000/metrics"

def check_vllm_status():
    try:
        response = requests.get(VLLM_METRICS_URL)
        data = response.text
        
        # 解析你需要的数据，vLLM 返回的是 Prometheus 格式
        metrics = {}
        for line in data.split('\n'):
            if line.startswith("#") or not line: continue
            
            # 示例：抓取正在运行的请求数
            if "vllm:num_requests_running" in line:
                metrics["running_reqs"] = float(line.split()[-1])
            
            # 示例：抓取 KV Cache 使用率 (显存相关)
            if "vllm:gpu_cache_usage_perc" in line:
                metrics["gpu_cache"] = float(line.split()[-1])
                
            # 示例：抓取 Token 生成速度
            if "vllm:avg_generation_throughput_toks_per_s" in line:
                metrics["gen_speed"] = float(line.split()[-1])

        print(f"📊 vLLM Status: Running={metrics.get('running_reqs', 0)} | "
              f"GPU Cache={metrics.get('gpu_cache', 0)*100:.1f}% | "
              f"Speed={metrics.get('gen_speed', 0):.1f} tok/s")

    except Exception as e:
        print(f"无法连接 vLLM: {e}")

# 你可以在 LangGraph 跑任务的时候，单独循环调用这个函数
check_vllm_status()