# RAG 系统测试框架

## 📋 概述

完整的 RAG 系统测试框架，包含三个独立的测试套件：

1. **vLLM Baseline** - 纯推理性能基线（无检索）
2. **RAG System** - 系统级性能测试（延迟、吞吐量、多跳）
3. **Answer Quality** - 答案质量评估（EM/F1）

## 🎯 测试理念

遵循 **两层测试方法**（CLAUDE.md 要求）：

```
第 1 层: vLLM-Only Baseline（设备特性描述）
         ↓
         建立性能边界
         ↓
第 2 层: RAG End-to-End（工作流行为）
         ↓
         通过对比基线来解释开销
```

**关键原则：**
- ✅ 区分 vLLM-only vs RAG end-to-end
- ✅ 报告 P95/P99 尾延迟，不仅仅是平均值
- ✅ 使用基线来解释 RAG 结果
- ✅ 关注系统行为（系统性能）和答案质量（分开测试）

## 📁 目录结构

```
tests/
├── README.md                    # 本文件 - 测试框架概述
│
├── vllm_baseline/               # vLLM 基线测试
│   ├── README.md                # vLLM 基线详细说明
│   ├── vllm_baseline.py         # 核心 benchmark
│   ├── visualize_vllm_baseline.py  # 可视化工具
│   └── test_vllm_baseline.py    # 自动化脚本 ⭐
│
├── rag_system/                  # RAG 系统性能测试与答案质量评估
│   ├── README.md                # RAG 系统测试详细说明
│   ├── system_bench.py          # 系统性能 benchmark
│   ├── bench_hotpotqa_fullwiki.py  # HotpotQA 答案质量 benchmark ⭐
│   ├── visualize.py             # 可视化工具
│   └── test_hop2rag.py          # Hop2Rag 自动化脚本 ⭐
│
├── data/hotpotqa/               # HotpotQA 数据集与索引脚本
│   ├── index_hotpotqa_fullwiki.py   # 索引脚本
│   ├── verify_hotpotqa_setup.py     # 验证脚本
│   ├── guide.md                     # HotpotQA 使用指南
│   └── EMBEDDING_MODELS.md          # 嵌入模型配置说明
│
├── results/                     # 所有测试结果
│   ├── vllm_baseline/          # vLLM 基线结果
│   ├── rag_system/             # RAG 系统测试结果
│   └── plots/                  # 图表
│
└── run_interactive.py          # 交互式运行工具
```

## 🚀 快速开始（30 秒）

```bash
# 步骤 1: 建立 vLLM 基线（重要：先做这个！）
cd vllm_baseline
python test_vllm_baseline.py --context-length 2000 --num-requests 50

# 步骤 2: 测试 Hop2Rag 实现（10 个问题）
cd ../rag_system
python test_hop2rag.py --limit 10 --k 20

# 步骤 3: 查看结果
cd ../results/plots && ls *.png
cd ../vllm_baseline/plots && ls *.png
```

## 📊 测试类型对比

| 维度 | vLLM Baseline | RAG System Performance | Answer Quality |
|------|--------------|----------------------|----------------|
| **测试对象** | 纯 LLM 推理 | 完整 RAG 工作流（系统性能） | 答案准确性 |
| **输入** | 合成文本 | HotpotQA 问题 | HotpotQA 问题 |
| **包含检索** | ❌ 否 | ✅ 是 | ✅ 是 |
| **关键指标** | P50/P95/P99 延迟 | 延迟 + 跳数 + 上下文 | EM/F1 |
| **用途** | 建立性能基线 | 评估系统行为 | 评估答案质量 |
| **推荐工具** | `test_vllm_baseline.py` | `system_bench.py` | `bench_hotpotqa_fullwiki.py` |
| **输出位置** | `results/vllm_baseline/` | `results/rag_system/` | `results/` |

## 🔄 完整测试工作流

### 步骤 0: 准备环境

```bash
# 安装依赖
pip install numpy matplotlib seaborn

# 索引 HotpotQA 数据集（如果还没有）
cd ../scripts
python index_hotpotqa_fullwiki.py
cd ../tests
```

### 步骤 1: 建立 vLLM 基线

**目的：** 描述纯推理性能，不包含任何检索开销。

```bash
cd vllm_baseline

# 快速基线
python test_vllm_baseline.py --context-length 2000 --num-requests 50

# 完整基线（推荐）
python test_vllm_baseline.py --context-sweep --num-requests 100

# 同时测试并发
python test_vllm_baseline.py --context-sweep --test-concurrency --num-requests 100
```

**输出：**
- 延迟 vs 上下文长度曲线
- 吞吐量 vs 上下文长度
- P95/P99 尾延迟行为

**详细说明：** 查看 [`vllm_baseline/README.md`](vllm_baseline/README.md)

### 步骤 2: 测试 RAG 工作流

使用**相同的工作负载参数**测试不同的 RAG 工作流。

```bash
cd ../rag_system

# Hop2Rag（自动化脚本）⭐ 推荐
python test_hop2rag.py --limit 50 --k 20

# 或者手动测试各个工作流
python system_bench.py --rag hop2rag --limit 50 --retrieval-k 20 \
    --out ../results/rag_system/hop2rag_traces.json

python system_bench.py --rag agrag --limit 50 --retrieval-k 20 \
    --out ../results/rag_system/agrag_traces.json

python system_bench.py --rag crag --limit 50 --retrieval-k 20 \
    --out ../results/rag_system/crag_traces.json
```

**输出：**
- 端到端延迟分布
- 多跳执行模式
- 上下文增长分析

**详细说明：** 查看 [`rag_system/README.md`](rag_system/README.md)

### 步骤 3: 对比和分析

```bash
cd rag_system

# 生成工作流对比图
python visualize.py --compare \
    hop2rag=../results/rag_system/hop2rag_traces.json \
    agrag=../results/rag_system/agrag_traces.json \
    crag=../results/rag_system/crag_traces.json \
    --out-dir ../results/plots/comparison

# 量化检索开销
echo "=== vLLM 基线 (上下文=2000) ==="
cat ../results/vllm_baseline/c1/stats_ctx2000_c1.json | \
    jq '{p50: .latency_median, p95: .latency_p95, p99: .latency_p99}'

echo "=== Hop2Rag (平均上下文~4000) ==="
cat ../results/rag_system/hop2rag_traces.stats.json | \
    jq '{p50: .latency_median, p95: .latency_p95, p99: .latency_p99}'

# 开销 = RAG延迟 - vLLM基线延迟 = 检索开销 + 多跳协调开销
```

### 步骤 4: 评估答案质量

```bash
cd ../rag_system

# Hop2Rag 答案质量
python bench_hotpotqa_fullwiki.py --rag hop2rag --limit 100 --retrieval-k 20

# AgRag 基线
python bench_hotpotqa_fullwiki.py --rag agrag --limit 100 --retrieval-k 20

# 对比结果
cat ../results/hop2rag_bench.pretty.json | jq
cat ../results/agrag_bench.pretty.json | jq
```

**输出：**
- Answer EM/F1
- Supporting Facts EM/F1
- Joint 指标

**详细说明：** 查看 [`rag_system/README.md`](rag_system/README.md)

## 📈 预期结果示例

### vLLM Baseline

| 上下文长度 | P50 延迟 | P95 延迟 | P99 延迟 | 吞吐量 |
|-----------|---------|---------|---------|--------|
| 500       | 0.5s    | 0.6s    | 0.7s    | 15 r/s |
| 2000      | 1.0s    | 1.3s    | 1.5s    | 8 r/s  |
| 4000      | 1.8s    | 2.3s    | 2.7s    | 4 r/s  |

### RAG System Performance

| 工作流 | P50 延迟 | P95 延迟 | 平均跳数 | 平均上下文 | 开销 |
|--------|---------|---------|---------|-----------|------|
| vLLM Baseline | 1.0s | 1.3s | 0 | 2000 | - |
| AgRag | 1.5s | 2.0s | 1 | 3500 | +0.5s |
| Hop2Rag | 2.2s | 3.1s | 1.8 | 4500 | +1.2s |

### Answer Quality

| 工作流 | Answer EM | Answer F1 | SP F1 | Joint F1 |
|--------|-----------|-----------|-------|----------|
| AgRag | 58% | 67% | 52% | 45% |
| Hop2Rag | 65% | 72% | 63% | 55% |
| 改进 | **+7%** | **+5%** | **+11%** | **+10%** |

## 🎯 关键发现模板

完成所有测试后，可以得出以下结论：

1. **检索开销**：多跳增加 X 秒的检索开销
2. **上下文增长影响**：从 2K 到 4.5K tokens 的上下文增长增加 Y 秒
3. **尾延迟**：由于多跳中的尾部效应，P99 延迟增加 Z%
4. **答案质量改进**：Hop2Rag 的 Answer EM 提升 A%，SP F1 提升 B%

## ⚠️ 常见问题

### 1. 缺少依赖

```bash
# 错误: No module named 'matplotlib'
# 解决方案:
pip install matplotlib seaborn numpy
```

### 2. 没有 Chroma 数据库

```bash
# 错误: Chroma persist directory not found
# 解决方案: 索引 HotpotQA
cd ../scripts
python index_hotpotqa_fullwiki.py
```

### 3. vLLM 服务器未运行

```bash
# 错误: Connection refused to vLLM server
# 解决方案: 启动 vLLM 服务器
# 查看项目 README 或配置文件了解启动命令
```

### 4. 测试太慢

```bash
# 使用较小的测试规模
python test_vllm_baseline.py --context-length 2000 --num-requests 10
python test_hop2rag.py --limit 5 --k 10
python bench_hotpotqa_fullwiki.py --rag hop2rag --limit 10
```

## 📚 详细文档

- **vLLM Baseline**：[`vllm_baseline/README.md`](vllm_baseline/README.md)
- **RAG System Performance**：[`rag_system/README.md`](rag_system/README.md)
- **HotpotQA Setup Guide**：[`data/hotpotqa/guide.md`](data/hotpotqa/guide.md)
- **Embedding Models**：[`data/hotpotqa/EMBEDDING_MODELS.md`](data/hotpotqa/EMBEDDING_MODELS.md)

## 🔬 研究焦点

如 `CLAUDE.md` 所述，这些工具关注：

**系统级行为（system_bench + vllm_baseline）：**
- ✅ 延迟分布（P50/P95/P99）
- ✅ 上下文增长和内存压力
- ✅ 多跳执行模式
- ✅ 吞吐量和并发性

**答案质量（bench_hotpotqa）：**
- ✅ Answer EM/F1
- ✅ Supporting Facts 质量

**分离测试的原因：** 系统/架构研究（ISCA/MICRO/ASPLOS）vs NLP/ML 研究（EMNLP/ACL）需要不同的评估角度。

## 🎓 引用建议

如果使用这些测试工具进行研究：

> "我们采用两层测试方法：(1) vLLM-only 基线建立纯推理性能边界，
> (2) RAG 端到端测试评估完整工作流。通过对比分析，我们量化了
> 检索开销和多跳协调成本。所有测试在相同的 HotpotQA + FullWiki
> 设置上进行，确保可控和可重现的分析。"

---

**版本：** v1.0
**最后更新：** 2024-01-19
**维护者：** RAG 系统测试团队
