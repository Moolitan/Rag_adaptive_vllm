# 修复更新日志 (CHANGELOG)

## [2026-01-19] - 代码质量与可维护性改进

### 🔴 严重问题修复

#### 修复 `write_jsonl` 函数重复定义
- **文件**: `Agrag/tests/cores.py`
- **问题**: 同时导入和定义了 `write_jsonl`，导致命名冲突
- **修复**: 移除了从 `core.logging` 的导入，保留本地定义
- **影响**: 消除了潜在的 `TypeError` 风险

#### 修正路径解析注释
- **文件**:
  - `Agrag/tests/data/hotpotqa/index_hotpotqa_fullwiki.py`
  - `Agrag/tests/data/hotpotqa/verify_hotpotqa_setup.py`
- **问题**: 注释说路径是 `parents[2]` 但应该是 `parents[3]`
- **修复**: 更新注释以反映实际目录结构
- **影响**: 消除误导性文档

---

### 🟠 高优先级改进

#### 新增：集中式嵌入模型配置
- **新增类**: `EmbeddingConfig` in `Agrag/core/config.py`
- **功能**:
  - 自动路径检测（环境变量 → 本地路径 → HuggingFace Hub）
  - 统一嵌入模型配置
  - 支持环境变量覆盖
- **代码行数减少**: 从 40+ 行减少到 3 行

#### 迁移到集中配置
- **修改的文件**:
  - `Agrag/tests/data/hotpotqa/index_hotpotqa_fullwiki.py`
  - `Agrag/tests/data/hotpotqa/verify_hotpotqa_setup.py`
- **改进**:
  - 消除了硬编码路径
  - 确保索引和验证使用相同模型
  - 提高代码可移植性

---

### 🟡 中等优先级改进

#### 修复文档引用
- **文件**: `Agrag/tests/vllm_baseline/README.md`
- **修复**: 删除了 3 个不存在文件的引用
  - `docs/RAG_PROMPT_FORMAT.md`
  - `docs/BEFORE_AFTER_COMPARISON.md`
  - `docs/IMPLEMENTATION_NOTES.md`

#### 更新主 README
- **文件**: `Agrag/tests/README.md`
- **改进**:
  - 修正目录结构说明（移除 `answer_quality/` 目录）
  - 更新文件路径（`rag_system/bench_hotpotqa_fullwiki.py`）
  - 修正测试类型表格
  - 更新所有文档链接

---

### 📝 新增文档

#### 检查报告
- **文件**: `docs/Agrag_tests_检查报告.md`
- **内容**: 完整的代码审查报告，包含：
  - 8 个问题的详细分析
  - 具体修复建议
  - 代码示例

#### 修复总结
- **文件**: `docs/修复总结.md`
- **内容**: 所有修复的详细说明和验证结果

---

### ✅ 验证

- [x] Python 语法检查通过
- [x] 路径计算验证通过
- [x] 导入逻辑验证通过
- [x] 文档链接检查通过

---

### 📊 统计

- **修复的问题**: 8 个（6 个已修复，2 个低优先级待处理）
- **修改的文件**: 5 个
- **新增的文件**: 3 个（包括文档）
- **代码行数减少**: ~80 行

---

### 🚀 如何使用新配置

```bash
# 默认使用（自动检测本地模型）
python tests/data/hotpotqa/index_hotpotqa_fullwiki.py ...

# 使用环境变量自定义
export EMBEDDING_MODEL_PATH="/your/custom/path"
python tests/data/hotpotqa/index_hotpotqa_fullwiki.py ...
```

---

### 💡 后续改进建议

#### 立即可做
- [ ] 在 `.env.example` 中添加 `EMBEDDING_MODEL_PATH` 示例
- [ ] 运行完整测试验证修复

#### 未来改进
- [ ] 为 `EmbeddingConfig` 添加单元测试
- [ ] 统一类型注解风格（使用 Python 3.10+ 语法）
- [ ] 填充或删除空的 `docs/` 目录
- [ ] 创建配置验证脚本

---

**修复者**: Claude Code
**日期**: 2026-01-19
**版本**: v1.0
