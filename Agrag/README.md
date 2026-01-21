# Agrag

基于 LangGraph 的自适应 RAG 框架，支持多种 RAG 变体（CRAG、Adaptive RAG、Self-RAG、Hop2RAG），集成 vLLM 本地推理和向量数据库检索，适用于 HotpotQA 等多跳问答任务。

![工作流图示](agent_workflow.png)

## 目录结构

```
Agrag/
├── core/                    # 核心组件
│   ├── chains.py            # LLM 链路定义
│   ├── config.py            # 全局配置（自动加载 .env）
│   ├── logging.py           # 日志工具
│   ├── prompt_manager.py    # 提示词管理
│   └── vectorstores.py      # 向量库/Embeddings
├── graph/                   # LangGraph 工作流
│   ├── state.py             # 状态类型定义
│   └── rag/                 # RAG 变体实现
│       ├── ag_rag.py        # Adaptive RAG
│       ├── c_rag.py         # Corrective RAG
│       ├── hop2_rag.py      # Hop2RAG (多跳检索)
│       ├── self_rag.py      # Self-RAG
│       └── factories/       # 节点/边工厂
├── runner/                  # 应用构建器
│   └── engine.py            # 构建/运行 LangGraph 应用
├── scripts/                 # 启动脚本
│   ├── run_hotpotqa.sh      # HotpotQA 测试脚本
│   └── run_interactive.sh   # 交互式测试脚本
├── tests/                   # 测试与评估
│   ├── bench_hotpotqa_fullwiki.py  # HotpotQA fullwiki 基准测试
│   ├── run_interactive.py          # 交互式问答测试
│   └── HotpotQA/                   # HotpotQA 数据与评估脚本
├── studio.py                # LangGraph Studio 入口
├── langgraph.json           # LangGraph Studio 配置
├── .env                     # 环境变量配置（不上传 git）
├── .env.example             # 环境变量模板（上传 git）
└── log/                     # 日志输出目录
```

## RAG 变体

| 变体 | 描述 | 适用场景 |
|------|------|----------|
| `crag` | Corrective RAG，检索后进行相关性判断，不相关时使用 Web 搜索 | 需要纠错的场景 |
| `agrag` | Adaptive RAG，根据问题类型自动路由（向量库/Web） | 通用问答 |
| `selfrag` | Self-RAG，生成后进行幻觉检测和答案验证 | 需要高可靠性的场景 |
| `hop2rag` | Hop2RAG，支持多跳检索和自我反思 | 多跳推理问答（如 HotpotQA） |

## 工作流概要

1. **analyze_question**: 路由决定 `web_search` / `vectorstore`
2. **retrieve**: 向量库检索（支持 HotpotQA 或自定义 Chroma 集合）
3. **grade_docs**: 文档相关性打分；若为空或不相关则转向 `web_search`
4. **web_search**: Tavily 检索，累计 `web_search_rounds`
5. **generate**: 基于上下文生成答案并抽取 supporting facts
6. **grade_generation**: 幻觉检测 + 答案验证，不通过则重试
7. **finalize**: 结束运行

---

## 快速开始

### 1. 安装依赖

```bash
conda activate langgraph_vllm

# 核心依赖
pip install langchain langgraph langchain-community chromadb
pip install langchain-tavily       # Web 搜索
pip install vllm                   # 本地 LLM 推理
pip install python-dotenv          # 环境变量加载
pip install langsmith              # 调试监控
pip install "langgraph-cli[inmem]" # LangGraph Studio
```

### 2. 配置环境变量

从模板创建 `.env` 文件：

```bash
cp .env.example .env
```

编辑 `.env` 文件填入实际值：

```bash
# 向量数据库配置（必填）
AGRAG_PERSIST_DIR=/path/to/chroma_db
AGRAG_COLLECTION_NAME=your_collection_name

# Tavily API Key（Web 搜索，从 https://tavily.com/ 获取）
TAVILY_API_KEY=your-tavily-api-key

# LangSmith 配置（从 https://smith.langchain.com/ 获取）
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-api-key
LANGCHAIN_PROJECT=hotpotqa-debug
```

> **注意**：`.env` 文件会被自动加载，无需手动 `export` 环境变量。`.env` 已在 `.gitignore` 中，不会上传到 git。

### 3. 启动 vLLM 服务

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /mnt/Large_Language_Model_Lab_1/模型/models/Qwen-Qwen2.5-7B-Instruct \
    --served-model-name Qwen2.5 \
    --dtype auto \
    --api-key EMPTY \
    --port 8000
```

### 4. 运行测试

```bash
cd /home/wsh/Langgraph_research/Rag_adaptive_vllm/Agrag

# 交互式问答
python tests/run_interactive.py --rag hop2rag

# HotpotQA 基准测试
python tests/bench_hotpotqa_fullwiki.py \
    --rag hop2rag \
    --limit 20 \
    --retrieval-k 10 \
    --soft-match \
    --save-graph
```

---

## LangSmith 调试与监控

[LangSmith](https://smith.langchain.com/) 是 LangChain 官方的可观测性平台，可用于调试、评估和监控 LangGraph 应用。

### 获取 API Key

1. 访问 https://smith.langchain.com/ 注册/登录
2. 点击右上角设置图标 → API Keys → Create API Key
3. 复制 API Key

### 配置

在 `.env` 文件中设置：

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-api-key
LANGCHAIN_PROJECT=hotpotqa-debug
```

### 查看 Traces

配置完成后，运行任何测试都会自动将 traces 发送到 LangSmith。

**Web 界面查看：**
1. 访问 https://smith.langchain.com/
2. 在左侧栏选择项目（如 `hotpotqa-debug`）
3. 点击 **Runs** 查看所有执行记录
4. 点击具体的 run 查看详情

**可以看到：**
- 完整的调用链路图
- 每个节点的输入/输出
- LLM 的 prompt 和 response
- 执行耗时和 token 使用量
- 错误信息和堆栈

**Python SDK 查看：**

```python
from langsmith import Client

client = Client()
runs = client.list_runs(project_name="hotpotqa-debug", limit=10)
for run in runs:
    print(f"Run: {run.id}, Status: {run.status}")
    print(f"  Inputs: {run.inputs}")
    print(f"  Outputs: {run.outputs}")
```

---

## LangGraph Studio 可视化调试

[LangGraph Studio](https://github.com/langchain-ai/langgraph-studio) 是 LangGraph 的可视化调试工具，支持实时查看工作流执行、单步调试和交互式测试。

### 安装

```bash
pip install "langgraph-cli[inmem]"
```

### 配置文件说明

项目已包含必要的配置文件：

**langgraph.json** - Studio 配置：
```json
{
  "python_version": "3.11",
  "dependencies": ["."],
  "graphs": {
    "hop2rag": "./studio.py:hop2rag",
    "crag": "./studio.py:crag",
    "agrag": "./studio.py:agrag",
    "selfrag": "./studio.py:selfrag"
  },
  "env": ".env"
}
```

**.env** - 环境变量（必须配置，参考 `.env.example`）

### 启动 Studio

#### 步骤 1：启动 API 服务器

```bash
cd /home/wsh/Langgraph_research/Rag_adaptive_vllm/Agrag
langgraph dev --host 0.0.0.0 --port 8123
```

启动成功后会显示：
```
Welcome to LangGraph!

- 🚀 API: http://0.0.0.0:8123
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://0.0.0.0:8123
- 📚 API Docs: http://0.0.0.0:8123/docs
```

#### 步骤 2：获取服务器 IP

```bash
hostname -I
# 例如输出：192.168.0.60
```

#### 步骤 3：访问 Studio UI

**重要**：直接访问 `http://服务器IP:8123` 只会返回 `{"ok": true}`，这是 API 健康检查响应。

要打开可视化界面，需要通过 LangSmith 托管的 Studio UI 访问：

```
https://smith.langchain.com/studio/?baseUrl=http://服务器IP:8123
```

例如：
```
https://smith.langchain.com/studio/?baseUrl=http://192.168.0.60:8123
```

### 使用 Studio

1. **选择 Graph**：在左上角下拉菜单选择 RAG 变体（hop2rag、crag、agrag、selfrag）

2. **查看工作流图**：中间区域显示工作流的节点和边

3. **运行测试**：
   - 点击 "New Thread" 创建新会话
   - 在输入框输入问题，如：`Were Scott Derrickson and Ed Wood of the same nationality?`
   - 点击 "Submit" 执行

4. **查看执行过程**：
   - 实时看到每个节点的执行状态
   - 点击节点查看输入/输出详情
   - 查看 LLM 调用的完整 prompt 和 response

5. **调试功能**：
   - 单步执行：逐节点执行查看中间状态
   - 断点：在特定节点暂停执行
   - 重放：重新执行某个节点

### 远程服务器访问方式

#### 方式一：局域网直接访问（推荐）

如果笔记本和服务器在同一局域网：

1. 服务器启动 Studio：`langgraph dev --host 0.0.0.0 --port 8123`
2. 笔记本浏览器访问：`https://smith.langchain.com/studio/?baseUrl=http://服务器IP:8123`

#### 方式二：SSH 端口转发

适用于无法直接访问服务器 IP 的情况：

```bash
# 在笔记本终端执行
ssh -L 8123:localhost:8123 wsh@服务器IP

# 然后访问
https://smith.langchain.com/studio/?baseUrl=http://localhost:8123
```

### 常见问题

**Q: 访问 `http://服务器IP:8123` 只显示 `{"ok": true}`？**

A: 这是正常的 API 健康检查响应。Studio UI 需要通过 `https://smith.langchain.com/studio/?baseUrl=http://服务器IP:8123` 访问。

**Q: 启动报错 `Missing required environment variable: AGRAG_PERSIST_DIR`？**

A: 检查 `.env` 文件是否存在且包含 `AGRAG_PERSIST_DIR` 配置。可以从模板创建：`cp .env.example .env`

**Q: Studio UI 无法连接到 API？**

A: 检查：
1. 服务器防火墙是否开放 8123 端口：`sudo ufw allow 8123`
2. 服务器和笔记本是否在同一网络
3. API 服务器是否正常运行（访问 `http://服务器IP:8123` 应返回 `{"ok": true}`）

---

## 命令行参数

### bench_hotpotqa_fullwiki.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--rag` | RAG 变体：`crag`, `agrag`, `selfrag`, `hop2rag` | `hop2rag` |
| `--limit` | 测试样本数量限制 | `100` |
| `--retrieval-k` | 向量库检索 top-k | `10` |
| `--soft-match` | 使用宽松匹配评估 | `false` |
| `--save-graph` | 保存工作流图示 | `false` |
| `--hotpot-json` | HotpotQA 数据文件路径 | `tests/HotpotQA/data/hotpot_dev_fullwiki_v1.json` |
| `--out` | 输出文件路径 | `tests/results/bench_hotpotqa_fullwiki.jsonl` |

### run_interactive.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--rag` | RAG 变体：`crag`, `agrag`, `selfrag`, `hop2rag` | `crag` |
| `--save-graph` | 保存工作流图示 | `false` |

> **注意**：向量数据库路径（`AGRAG_PERSIST_DIR`）和集合名（`AGRAG_COLLECTION_NAME`）现在从 `.env` 文件自动读取，无需通过命令行参数指定。

## 输出文件

| 文件 | 说明 |
|------|------|
| `tests/results/bench_hotpotqa_fullwiki.jsonl` | 测试结果（每行一个样本） |
| `tests/results/bench_hotpotqa_fullwiki.pretty.json` | 测试结果（格式化） |
| `agent_workflow.png` | 工作流图示 |
| `log/` | 运行日志 |
