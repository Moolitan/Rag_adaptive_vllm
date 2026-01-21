# vLLM Baseline 性能测试

## 📋 目的

建立 **纯推理性能基线**（不包含检索和 RAG 逻辑），用于：
- 量化 RAG 系统的检索开销
- 识别 vLLM 瓶颈（上下文长度、并发）
- 理解性能区域（计算限制 vs 内存限制）

## 🎯 测量内容

- ✅ 纯 vLLM 推理延迟（无检索开销）
- ✅ TTFT, ITL, E2E 延迟（P50/P95/P99）
- ✅ 上下文长度扩展（500 → 4000+ tokens）
- ✅ 并发扩展（1 → 8+ 并发请求）
- ✅ 吞吐量限制

## 🔧 工具说明

### 1. `vllm_baseline.py` - 核心 Benchmark

生成 **RAG 风格的提示词**（短问题 + 长文档），直接调用 vLLM 进行推理测试。

**特点：**
- 使用 **短问题（~20 tokens）+ 多样化长文档（500-16000 tokens）**
- 模拟真实 RAG 场景：用户查询 + 检索文档
- 每个请求生成唯一的文档组合，避免 KV cache 污染
- 不访问向量数据库，不涉及检索

**单独运行：**
```bash
# 测试单一上下文长度
python vllm_baseline.py --context-length 2000 --num-requests 50

# 扫描多个上下文长度（推荐）⭐
python vllm_baseline.py --context-sweep 500,1000,2000,4000,8000,16000 --num-requests 100

# 测试并发
python vllm_baseline.py --context-length 2000 --concurrency 4 --num-requests 50
```

### 2. `visualize_vllm_baseline.py` - 可视化工具

从已保存的测试数据生成性能图表。

**特点：**
- 可独立运行，不重新测试
- 支持从单个 trace 文件或 summary 文件读取
- 生成延迟 vs 上下文、吞吐量 vs 上下文、CDF 等图表

**单独运行：**
```bash
# 从 summary 生成对比图
python visualize_vllm_baseline.py --summary ../results/vllm_baseline/summary_all.json \
                                    --out-dir ../results/vllm_baseline/plots

# 从单个 trace 生成 CDF
python visualize_vllm_baseline.py --trace ../results/vllm_baseline/c1/traces_ctx2000_c1.json \
                                    --out-dir ../results/vllm_baseline/plots
```

### 3. `test_vllm_baseline.py` - 自动化脚本 ⭐ 推荐

一键运行完整测试流程：运行 benchmark → 生成图表。

**特点：**
- 自动调用 `vllm_baseline.py` 和 `visualize_vllm_baseline.py`
- 支持上下文扫描（500-16000 tokens）和并发测试
- 自动整合多个测试结果

**推荐用法：**
```bash
# 快速测试（单一上下文）
python test_vllm_baseline.py --context-length 2000 --num-requests 50

# 完整扫描（推荐）⭐ 测试 500, 1000, 2000, 4000, 8000, 16000 tokens
python test_vllm_baseline.py --context-sweep --num-requests 100

# 同时测试并发
python test_vllm_baseline.py --context-sweep --test-concurrency --num-requests 100

# 跳过画图（只收集数据）
python test_vllm_baseline.py --context-sweep --num-requests 50 --skip-plots
```

## 📂 输出文件

```
tests/results/vllm_baseline/
├── c1/                              # 并发=1 的结果
│   ├── traces_ctx500_c1.json       # 500 tokens 请求追踪
│   ├── stats_ctx500_c1.json        # 500 tokens 统计数据
│   ├── traces_ctx1000_c1.json
│   ├── stats_ctx1000_c1.json
│   ├── traces_ctx2000_c1.json
│   ├── stats_ctx2000_c1.json
│   ├── traces_ctx4000_c1.json
│   ├── stats_ctx4000_c1.json
│   ├── traces_ctx8000_c1.json      # 新增：8K tokens
│   ├── stats_ctx8000_c1.json
│   ├── traces_ctx16000_c1.json     # 新增：16K tokens
│   └── stats_ctx16000_c1.json
├── summary_all.json                 # 所有运行的汇总
└── plots/
    ├── latency_vs_context_c1.png   # 延迟扩展曲线（P50/P95/P99）⭐ 最重要
    ├── throughput_vs_context_c1.png # 吞吐量扩展曲线
    └── latency_cdf_ctx*.png         # 各上下文长度的 CDF
```

## 📊 生成的图表

1. **`latency_vs_context_c1.png`** ⭐ 最重要
   - 显示 P50/P95/P99 延迟如何随上下文长度增长
   - 用于识别内存限制区域（非线性增长）

2. **`throughput_vs_context_c1.png`**
   - 显示吞吐量如何随上下文长度下降
   - 用于找到最大可持续吞吐量

3. **`latency_cdf_ctx*.png`**
   - 显示特定上下文长度的尾延迟分布
   - 用于分析 P95/P99 尾部行为

## 🔬 如何使用基线

### 与 RAG 系统对比

```bash
# 步骤 1: 建立基线
cd tests/vllm_baseline
python test_vllm_baseline.py --context-sweep --num-requests 100

# 步骤 2: 运行 RAG 测试
cd ../rag_system
python test_hop2rag.py --limit 50 --k 20

# 步骤 3: 对比分析
# 查看 vLLM 基线 (context=2000)
cat ../results/vllm_baseline/c1/stats_ctx2000_c1.json | jq '{p50: .latency_median, p95: .latency_p95}'

# 查看 Hop2Rag (平均 context~4000)
cat ../results/rag_system/hop2rag_traces.stats.json | jq '{p50: .latency_median, p95: .latency_p95}'

# 计算开销
# RAG_latency - vLLM_baseline = 检索开销 + 多跳协调开销
```

### 预期结果示例

| 上下文长度 | P50 延迟 | P95 延迟 | P99 延迟 | 吞吐量 | 备注 |
|-----------|---------|---------|---------|--------|------|
| 500       | 0.5s    | 0.6s    | 0.7s    | 15 r/s | 计算受限 |
| 1000      | 0.7s    | 0.9s    | 1.0s    | 12 r/s | 计算受限 |
| 2000      | 1.0s    | 1.3s    | 1.5s    | 8 r/s  | 计算受限 |
| 4000      | 1.8s    | 2.3s    | 2.7s    | 4 r/s  | 计算受限 |
| 8000      | 3.5s    | 4.5s    | 5.2s    | 2 r/s  | 过渡区 |
| 16000     | 7.5s    | 10.0s   | 12.0s   | 1 r/s  | 内存受限 ⚠️ |

**观察：**
- 500-4000: 延迟线性增长 → 计算受限（正常）
- 8000-16000: 延迟非线性增长 → 内存瓶颈开始显现
- P99 在 16K 显著恶化 → GPU 内存压力

## ⚠️ 注意事项

1. **RAG 风格 prompt**
   - 基线使用 **短问题（~20 tokens）+ 多样化长文档（可变）**
   - 模拟真实 RAG 场景：用户查询 + 检索结果
   - 每个请求的文档组合不同，避免 KV cache 污染
   - 详细格式说明：查看 [docs/RAG_PROMPT_FORMAT.md](docs/RAG_PROMPT_FORMAT.md)

2. **上下文长度**
   - `--context-length` 参数指定**文档长度**（不含问题和格式文本）
   - 实际 prompt 总长度 ≈ 文档长度 + 60 tokens（问题 + 格式）
   - 例如：`--context-length 2000` → 总 prompt ~2060 tokens

3. **上下文长度估算**
   - 使用经验法则：~4 字符/token
   - 实际 token 数可能略有偏差（±10%）

4. **预热 vLLM**
   - 首次请求可能较慢（加载模型）
   - 建议运行至少 50 个请求以获得稳定结果

5. **依赖项**
   ```bash
   pip install numpy matplotlib seaborn
   ```

## 📖 进一步阅读

- 完整测试工作流：查看 `tests/README.md`
- RAG 系统测试：查看 `tests/rag_system/README.md`
- 答案质量测试：查看 `tests/answer_quality/README.md`
