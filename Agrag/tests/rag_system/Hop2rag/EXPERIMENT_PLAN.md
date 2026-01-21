# Agentic RAG 系统层面优化实验计划

## 背景

在保留 Agentic RAG 的**反思、纠偏与循环机制**的前提下，从系统层面观察和优化推理效率。关注 GPU 内存、KV Cache、数据流调度等底层行为。

---

## vLLM 启动配置

不同实验需要不同的 vLLM 启动参数。以下是各实验对应的配置：

### 基础配置（所有实验通用）

```bash
# 基础启动命令
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9
```

### 实验专用配置

| 实验 | 额外参数 | 说明 |
|------|----------|------|
| 实验一：KV Cache 分析 | 无需 vLLM | 仅模拟 prompt 分析 |
| 实验二：GPU 内存时序 | `--disable-log-requests` | 减少日志干扰 |
| 实验三：流水线重叠 | `--disable-log-requests` | 减少日志干扰 |
| 实验四：vLLM 指标 | `--enable-prefix-caching` | **关键：启用 prefix caching** |
| 实验五：上下文增长 | `--max-model-len 8192` | 可调整最大上下文长度 |
| 实验六：并发测试 | `--max-num-seqs 32` | 调整最大并发序列数 |

### 详细配置说明

#### 实验四专用：对比 Prefix Caching 效果

```bash
# 配置 A：不启用 prefix caching（baseline）
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/Qwen2.5-7B-Instruct \
    --port 8000 \
    --disable-log-requests

# 配置 B：启用 prefix caching
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/Qwen2.5-7B-Instruct \
    --port 8000 \
    --enable-prefix-caching \
    --disable-log-requests
```

#### 实验六专用：并发性能测试

```bash
# 低并发配置
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/Qwen2.5-7B-Instruct \
    --port 8000 \
    --max-num-seqs 8 \
    --disable-log-requests

# 高并发配置
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/Qwen2.5-7B-Instruct \
    --port 8000 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 8192 \
    --disable-log-requests
```

#### 高级配置：内存优化实验

```bash
# 启用 chunked prefill（减少内存峰值）
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/Qwen2.5-7B-Instruct \
    --port 8000 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 4096 \
    --disable-log-requests

# 启用 prefix caching + chunked prefill
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/Qwen2.5-7B-Instruct \
    --port 8000 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --disable-log-requests
```

---

## 核心观察维度

```
┌─────────────────────────────────────────────────────────────────┐
│                    Agentic RAG 系统层面观察                       │
├─────────────────────────────────────────────────────────────────┤
│  1. KV Cache 行为                                                │
│     - 多轮 LLM 调用间的 prefix 重叠率                             │
│     - Cache 命中/未命中模式                                       │
│     - 循环迭代中的 cache 增长曲线                                 │
│                                                                 │
│  2. GPU 内存动态                                                 │
│     - 检索 vs 生成阶段的显存占用模式                              │
│     - 峰值内存出现位置                                           │
│     - 内存碎片化情况                                             │
│                                                                 │
│  3. 计算/IO 重叠                                                 │
│     - 检索（CPU/IO密集）与生成（GPU密集）的时间重叠可能性          │
│     - Embedding 计算与 LLM 推理的并行机会                         │
│                                                                 │
│  4. 请求级行为模式                                               │
│     - 不同问题类型的跳数分布                                      │
│     - 循环终止条件触发时机                                        │
│     - 上下文长度增长模式                                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 实验一：KV Cache 复用潜力分析

### 目标
分析 Agentic RAG 中多次 LLM 调用之间的 prompt prefix 重叠程度，评估 KV Cache 复用的理论上限。

### vLLM 配置
**无需启动 vLLM**（纯 prompt 模拟分析）

### 运行命令
```bash
python tests/rag_system/Hop2rag/test_kv_cache_analysis.py \
    --limit 20 \
    --max-hops 3 \
    --docs-per-hop 10
```

### 观察点
```
Hop 0: [System Prompt] + [Question] + [Evidence_0] + [Instruction]
Hop 1: [System Prompt] + [Question] + [Evidence_0] + [Evidence_1] + [Instruction]
Hop 2: [System Prompt] + [Question] + [Evidence_0] + [Evidence_1] + [Evidence_2] + [Instruction]
       ↑_____________ 这部分是共享的，理论上可以复用 KV Cache _____________↑
```

### 输出
- `results/hop2rag/kv_cache/prefix_overlap_analysis.json`
- `results/hop2rag/kv_cache/prefix_overlap_heatmap.png`
- `results/hop2rag/kv_cache/token_breakdown.png`

---

## 实验二：GPU 内存时序分析

### 目标
记录一个完整请求生命周期中的 GPU 内存变化，识别内存峰值位置和空闲窗口。

### vLLM 配置
```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/Qwen2.5-7B-Instruct \
    --port 8000 \
    --disable-log-requests
```

### 运行命令
```bash
export AGRAG_PERSIST_DIR="/path/to/chroma/persist"
export AGRAG_COLLECTION_NAME="hotpot_fullwiki"

python tests/rag_system/Hop2rag/test_gpu_memory_profile.py \
    --limit 5 \
    --k 10 \
    --max-hops 3 \
    --sample-interval 10
```

### 期望观察到的模式
```
Memory (MB)
    ^
    |     ┌──┐ decompose    ┌──┐ extract     ┌──┐ decide
    |     │  │              │  │             │  │
    |  ───┘  └──────────────┘  └─────────────┘  └───── ...
    |     ↑                 ↑                 ↑
    |   LLM推理峰值       LLM推理峰值        LLM推理峰值
    |
    +──────────────────────────────────────────────────> Time
          ↑         ↑
        检索阶段   检索阶段
       (GPU空闲)  (GPU空闲)
```

### 输出
- `results/hop2rag/gpu_memory/memory_timeline.png`
- `results/hop2rag/gpu_memory/memory_by_node.png`
- `results/hop2rag/gpu_memory/memory_phases.png`

---

## 实验三：检索-生成流水线重叠分析

### 目标
分析检索（CPU/IO 密集）和生成（GPU 密集）操作的时间分布，评估流水线并行的可能性。

### vLLM 配置
```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/Qwen2.5-7B-Instruct \
    --port 8000 \
    --disable-log-requests
```

### 运行命令
```bash
python tests/rag_system/Hop2rag/test_pipeline_overlap.py \
    --limit 10 \
    --k 10 \
    --max-hops 3
```

### 分析场景
```
当前串行模式:
  retrieve_0 → decompose_0 → grade_0 → extract_0 → retrieve_1 → ...
       CPU        GPU         CPU        GPU          CPU

潜在并行模式:
  retrieve_0 → decompose_0 → grade_0 → extract_0 → ...
                   ↓ (预取)
              retrieve_1 (并行)
```

### 输出
- `results/hop2rag/pipeline/timeline_gantt.png`
- `results/hop2rag/pipeline/overlap_analysis.json`
- `results/hop2rag/pipeline/resource_utilization.png`

---

## 实验四：vLLM 服务端指标观测

### 目标
观察 vLLM 内部的 KV Cache 命中、batch 行为、preemption 等指标。

### vLLM 配置（关键：启用 prefix caching）
```bash
# 对比配置 A：不启用
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/Qwen2.5-7B-Instruct \
    --port 8000 \
    --disable-log-requests

# 对比配置 B：启用 prefix caching
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/Qwen2.5-7B-Instruct \
    --port 8000 \
    --enable-prefix-caching \
    --disable-log-requests
```

### 运行命令
```bash
python tests/rag_system/Hop2rag/test_vllm_metrics.py \
    --limit 20 \
    --k 10 \
    --max-hops 3 \
    --vllm-endpoint http://localhost:8000 \
    --sample-interval 100
```

### 测量指标
- `vllm:prefix_cache_hit_rate` - Prefix Cache 命中率
- `vllm:gpu_cache_usage_perc` - GPU Cache 使用率
- `vllm:num_preemptions_total` - 请求抢占次数
- `vllm:avg_prompt_throughput_toks_per_s` - Prompt 吞吐量
- `vllm:avg_generation_throughput_toks_per_s` - 生成吞吐量

### 输出
- `results/hop2rag/vllm_metrics/cache_metrics.png`
- `results/hop2rag/vllm_metrics/throughput.png`
- `results/hop2rag/vllm_metrics/request_states.png`

---

## 实验五：上下文增长影响分析

### 目标
测量随着跳数增加，累积的文档上下文如何影响延迟和内存。

### vLLM 配置
```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/Qwen2.5-7B-Instruct \
    --port 8000 \
    --max-model-len 8192 \
    --disable-log-requests
```

### 运行命令
```bash
python tests/rag_system/Hop2rag/test_context_growth.py \
    --limit 20 \
    --k 10 \
    --max-hops 5
```

### 关键问题
- 延迟增长是**线性**还是**超线性**（O(n²) due to attention）？
- KV Cache 内存增长模式？
- 是否有明显的"上下文长度墙"？

### 输出
- `results/hop2rag/context_growth/context_vs_latency.png`
- `results/hop2rag/context_growth/context_vs_memory.png`
- `results/hop2rag/context_growth/token_accumulation.png`

---

## 实验六：多请求并发资源竞争

### 目标
分析并发请求时的 GPU 内存和 vLLM batch 行为。

### vLLM 配置（按并发度调整）
```bash
# 低并发 (1-4)
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/Qwen2.5-7B-Instruct \
    --port 8000 \
    --max-num-seqs 16 \
    --disable-log-requests

# 高并发 (8+)
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/Qwen2.5-7B-Instruct \
    --port 8000 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 16384 \
    --disable-log-requests
```

### 运行命令
```bash
python tests/rag_system/Hop2rag/test_concurrent_requests.py \
    --num-requests 20 \
    --concurrency 1,2,4,8 \
    --k 10 \
    --max-hops 3
```

### 测量指标
- 吞吐量 vs 并发度
- 延迟分布 (P50, P95, P99)
- GPU 内存峰值
- 缩放效率

### 输出
- `results/hop2rag/concurrency/throughput_vs_concurrency.png`
- `results/hop2rag/concurrency/latency_vs_concurrency.png`
- `results/hop2rag/concurrency/memory_vs_concurrency.png`

---

## 实验执行顺序建议

```
1. 实验一：KV Cache 分析        [无需 vLLM，快速运行]
   └─→ 获取理论 prefix 重叠率上限

2. 实验二：GPU 内存时序         [需要 vLLM]
   └─→ 识别 GPU 空闲窗口

3. 实验四：vLLM 指标观测        [需要 vLLM + prefix caching]
   └─→ 验证实际 cache 命中率 vs 理论值

4. 实验三：流水线重叠分析       [需要 vLLM]
   └─→ 评估调度优化空间

5. 实验五：上下文增长分析       [需要 vLLM]
   └─→ 理解延迟增长模式

6. 实验六：并发测试             [需要 vLLM，高资源消耗]
   └─→ 评估生产环境行为
```

---

## 潜在优化方向（基于实验结果）

### 1. KV Cache 优化
- 启用 vLLM prefix caching
- 设计 prompt 结构以最大化 prefix 共享
- 跨请求的 cache 预热

### 2. 流水线调度
- 检索预取（在 LLM 推理时预取下一跳文档）
- CPU-GPU 异步执行
- 动态 batch 策略

### 3. 内存管理
- 分页 attention (PagedAttention)
- 动态内存分配
- 检索阶段的显存卸载

### 4. 上下文压缩
- 动态文档截断
- 重要性采样
- 渐进式上下文构建

---

## 文件结构

```
Agrag/tests/rag_system/Hop2rag/
├── EXPERIMENT_PLAN.md                 # 本文档
├── test_hop2rag_latency.py           # 基础延迟测试
├── test_kv_cache_analysis.py         # 实验一：KV Cache 分析
├── test_gpu_memory_profile.py        # 实验二：GPU 内存时序
├── test_pipeline_overlap.py          # 实验三：流水线重叠
├── test_vllm_metrics.py              # 实验四：vLLM 指标
├── test_context_growth.py            # 实验五：上下文增长
└── test_concurrent_requests.py       # 实验六：并发竞争
```

---

## 环境变量设置

所有需要实际运行 Hop2Rag 的实验都需要设置以下环境变量：

```bash
export AGRAG_PERSIST_DIR="/path/to/your/chroma/persist/directory"
export AGRAG_COLLECTION_NAME="hotpot_fullwiki"
```

---

## 快速验证

```bash
# 1. 先运行不需要 vLLM 的实验一
python tests/rag_system/Hop2rag/test_kv_cache_analysis.py --limit 5 --max-hops 2

# 2. 启动 vLLM（基础配置）
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/Qwen2.5-7B-Instruct \
    --port 8000 \
    --disable-log-requests &

# 3. 运行实验二（小规模）
python tests/rag_system/Hop2rag/test_gpu_memory_profile.py --limit 2 --max-hops 2

# 4. 重启 vLLM 启用 prefix caching
pkill -f vllm
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/Qwen2.5-7B-Instruct \
    --port 8000 \
    --enable-prefix-caching \
    --disable-log-requests &

# 5. 运行实验四
python tests/rag_system/Hop2rag/test_vllm_metrics.py --limit 10
```
