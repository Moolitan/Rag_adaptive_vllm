# CRag2 测试代码

基于网上找的成熟 Corrective RAG 代码改造，适配项目的 FAISS + vLLM 架构。

## 📁 文件说明

- `test_c_rag2_performance.py` - 基础性能测试脚本（15KB）
- `test_crag2_on_squad_dev.py` - SQuAD 数据集评估脚本（7.3KB）
- `run.sh` - 运行脚本（包含环境配置和命令示例）
- `README.md` - 本文档

## 🔧 环境准备

### 1. 启动 vLLM 服务

```bash
# 激活环境
conda activate langgraph_vllm

# 启动 vLLM 服务
python -m vllm.entrypoints.openai.api_server \
    --model /mnt/Large_Language_Model_Lab_1/模型/models/Qwen-Qwen2.5-7B-Instruct \
    --served-model-name Qwen2.5 \
    --enable-prefix-caching \
    --disable-log-requests \
    --dtype auto \
    --api-key EMPTY \
    --port 8000
```

### 2. FAISS 向量数据库配置

#### 方式一：直接加载（首次运行较慢）

```bash
# 设置环境变量
export AGRAG_FAISS_DIR="/mnt/Large_Language_Model_Lab_1/faiss_wiki_db"

# 直接运行测试（首次加载 FAISS 需要较长时间）
python tests/rag_system/c_rag2/test_c_rag2_performance.py --limit 5
```

**注意**：首次加载 FAISS 数据库可能需要 1-2 分钟，后续查询会快很多。

#### 方式二：使用 FAISS 服务（推荐，支持预热）

**优势**：
- 预先加载数据库，避免测试时等待
- 多个测试脚本可共享同一个 FAISS 实例
- 支持远程调用

**启动 FAISS 服务**：

```bash
# 前台运行（推荐调试时使用）
conda activate langgraph_vllm
AGRAG_FAISS_DIR="/mnt/Large_Language_Model_Lab_1/faiss_wiki_db" python -m Agrag.Rag.faiss_server

# 或后台运行（推荐正式测试时使用）
nohup env AGRAG_FAISS_DIR="/mnt/Large_Language_Model_Lab_1/faiss_wiki_db" python -m Agrag.Rag.faiss_server > faiss_server.log 2>&1 &

# 检查服务状态
curl http://127.0.0.1:5100/health

# 预期输出：{"status": "healthy", "faiss_loaded": true}
```

**查看 FAISS 服务日志**：

```bash
# 如果是后台运行
tail -f faiss_server.log

# 停止后台服务
pkill -f "faiss_server"
```

### 3. 设置 Tavily API Key（可选）

```bash
# Web search 功能需要（测试中默认跳过）
export TAVILY_API_KEY="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"
```

## 🚀 运行测试

### 基础性能测试

```bash
cd Agrag
export AGRAG_FAISS_DIR="/mnt/Large_Language_Model_Lab_1/faiss_wiki_db"
# 基础测试（5个问题）
python tests/rag_system/c_rag2/test_c_rag2_performance.py \
    --limit 5 \
    --monitor-interval 0.5 \
    --retrieval-k 10
    
# 基础测试（5个问题）
python tests/rag_system/c_rag2/test_c_rag2_datacollector.py \
    --limit 5 \
    --retrieval-k 10


# 详细输出模式
python tests/rag_system/c_rag2/test_c_rag2_performance.py \
    --limit 10 \
    --monitor-interval 0.5 \
    --verbose
```

**参数说明**：
- `--limit N`: 测试问题数量（默认 5）
- `--monitor-interval SECONDS`: vLLM 监控采样间隔（默认 0.5 秒）
- `--verbose`: 显示详细输出

### SQuAD 数据集评估

```bash
cd Agrag

# 使用 SQuAD 数据集测试
python tests/rag_system/c_rag2/test_crag2_on_squad_dev.py \
    --squad-dev tests/rag_system/c_rag/data/SQUAD-dev-v2.0.json \
    --start 100 \
    --limit 10 \
    --monitor-interval 0.5 \
    --verbose
```

**参数说明**：
- `--squad-dev PATH`: SQuAD 数据集路径
- `--start N`: 起始索引（默认 0）
- `--limit N`: 测试样本数量（默认 10）
- `--monitor-interval SECONDS`: vLLM 监控采样间隔（默认 0.5 秒）
- `--verbose`: 显示详细输出

## 📊 输出结果

### 结果文件位置

- **基础测试**: `tests/results/crag2_performance/`
- **SQuAD 测试**: `tests/results/crag2_squad/`

### 输出文件

1. **performance_results.json** - 详细的每个请求的性能数据
   ```json
   [
     {
       "question": "问题内容",
       "answer": "生成的答案",
       "total_latency_sec": 2.34,
       "llm_calls": 3,
       "total_llm_latency_sec": 1.89,
       "total_input_tokens": 1234,
       "total_output_tokens": 567,
       "node_executions": 5,
       "used_web_search": false,
       "records": [...]
     }
   ]
   ```

2. **performance_stats.json** - 统计摘要
   ```json
   {
     "total_requests": 10,
     "latency_mean": 2.45,
     "latency_p50": 2.30,
     "latency_p95": 3.20,
     "llm_latency_mean": 1.85,
     "llm_calls_mean": 3.2,
     "input_tokens_mean": 1250.5,
     "output_tokens_mean": 580.3,
     "web_search_rate": 0.1
   }
   ```

3. **vllm_metrics.csv** - vLLM 服务器指标（时间序列）

### 性能指标说明

| 指标 | 说明 |
|------|------|
| **total_latency_sec** | 总延迟时间（秒） |
| **llm_calls** | LLM 调用次数 |
| **total_llm_latency_sec** | LLM 总耗时（秒） |
| **total_input_tokens** | 输入 token 总数 |
| **total_output_tokens** | 输出 token 总数 |
| **node_executions** | 节点执行次数 |
| **used_web_search** | 是否使用了 web 搜索 |
| **EM** | 精确匹配率（仅 SQuAD） |
| **F1** | F1 分数（仅 SQuAD） |

## 🔄 工作流程

CRag2 的执行流程：

```
                    ┌─────────────┐
                    │   retrieve  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────────┐
                    │ grade_documents │
                    └──────┬──────────┘
                           │
                    ┌──────▼──────────┐
                    │     decide      │
                    └─────┬───┬───────┘
                          │   │
              ┌───────────┘   └──────────┐
              │                           │
    ┌─────────▼─────────┐      ┌─────────▼─────────┐
    │     generate      │      │ transform_query   │
    └───────────────────┘      └─────────┬─────────┘
                                          │
                               ┌──────────▼──────────┐
                               │    web_search       │
                               └──────────┬──────────┘
                                          │
                               ┌──────────▼──────────┐
                               │     generate        │
                               └─────────────────────┘
```

**节点说明**：
1. **retrieve**: 从 FAISS 向量库检索相关文档（k=15）
2. **grade_documents**: 使用 LLM 评估文档相关性
3. **decide**: 根据评分决定是否需要 web 搜索
4. **transform_query**: 优化查询语句
5. **web_search**: 执行 web 搜索（测试中跳过）
6. **generate**: 基于文档生成答案

## 🔍 与原始代码的对比

### ✅ 保留的部分（来自网上的成熟代码）

- 所有 LangGraph 工作流逻辑
- 所有提示词模板（完全不变）
- 文档评分和过滤逻辑
- Query transformation 逻辑
- Web search 集成逻辑
- 错误处理机制

### 🔄 修改的部分（适配项目架构）

| 组件 | 原始代码 | 改造后 |
|------|---------|--------|
| **向量库** | Qdrant | FAISS |
| **Embeddings** | OpenAI (text-embedding-3-small) | HuggingFace (BAAI-bge-base-en-v1.5) |
| **LLM** | Anthropic Claude (claude-sonnet-4-5) | vLLM (Qwen2.5) |
| **配置方式** | Streamlit session state | 环境变量 |
| **UI** | Streamlit Web UI | 命令行脚本 |

## ⚠️ 注意事项

### FAISS 加载时间

- **首次加载**: 1-2 分钟（取决于数据库大小）
- **后续查询**: 毫秒级
- **建议**: 使用 FAISS 服务预热，避免测试时等待

### 内存占用

- FAISS 数据库会占用较大内存（约 10-20GB）
- 确保服务器有足够的可用内存
- 可以使用 `free -h` 检查内存状态

### Web Search 功能

- 测试代码中默认跳过 web search（避免 API 调用）
- 如需启用，需要：
  1. 设置 `TAVILY_API_KEY` 环境变量
  2. 修改 `web_search()` 函数实现

### 检索参数

- 当前配置：`k=15`（检索 15 个文档）
- 可在 `get_retriever()` 函数中调整

## 🐛 故障排查

### 问题 1: FAISS 加载失败

```bash
# 检查环境变量
echo $AGRAG_FAISS_DIR

# 检查目录是否存在
ls -lh $AGRAG_FAISS_DIR

# 检查文件权限
ls -l $AGRAG_FAISS_DIR/index.faiss
```

### 问题 2: vLLM 连接失败

```bash
# 检查 vLLM 服务状态
curl http://localhost:8000/v1/models

# 检查端口占用
lsof -i :8000
```

### 问题 3: FAISS 服务无响应

```bash
# 检查服务状态
curl http://127.0.0.1:5100/health

# 查看日志
tail -f faiss_server.log

# 重启服务
pkill -f "faiss_server"
nohup env AGRAG_FAISS_DIR="/mnt/Large_Language_Model_Lab_1/faiss_wiki_db" python -m Agrag.Rag.faiss_server > faiss_server.log 2>&1 &
```

### 问题 4: 导入错误

```bash
# 确保在正确的目录
cd /home/wsh/Langgraph_research/Rag_adaptive_vllm/Agrag

# 检查 Python 路径
python -c "import sys; print('\n'.join(sys.path))"

# 激活正确的环境
conda activate rag_adaptive
```

## 📝 快速开始示例

```bash
# 1. 激活环境
conda activate rag_adaptive

# 2. 设置环境变量
export AGRAG_FAISS_DIR="/mnt/Large_Language_Model_Lab_1/faiss_wiki_db"

# 3. 启动 FAISS 服务（推荐，预热数据库）
cd /home/wjj/Rag_adaptive_vllm
nohup env AGRAG_FAISS_DIR="/mnt/Large_Language_Model_Lab_1/faiss_wiki_db" python -m Agrag.Rag.faiss_server > faiss_server.log 2>&1 &

# 4. 等待 FAISS 加载完成（1-2分钟）
tail -f faiss_server.log
# 看到 "FAISS database loaded" 后按 Ctrl+C

# 5. 检查服务状态
curl http://127.0.0.1:5100/health

# 6. 运行测试
cd /home/wsh/Langgraph_research/Rag_adaptive_vllm/Agrag
python tests/rag_system/c_rag2/test_c_rag2_performance.py --limit 5 --verbose

# 7. 查看结果
cat tests/results/crag2_performance/performance_stats.json
```

## 📚 相关文档

- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [FAISS 文档](https://github.com/facebookresearch/faiss)
- [vLLM 文档](https://docs.vllm.ai/)
- [SQuAD 数据集](https://rajpurkar.github.io/SQuAD-explorer/)

## 🤝 贡献

如有问题或建议，请联系项目维护者。
