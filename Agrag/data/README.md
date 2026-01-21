# Data Directory - 数据管理

本目录用于管理 RAG 系统所需的数据集、向量数据库索引和语料库。

## 📁 目录结构

```
data/
├── README.md                          # 本文档
├── download.sh                        # 数据下载脚本
├── index_hotpotqa_fullwiki.py         # HotpotQA 索引脚本（Chroma）
├── index_wiki_faiss.py                # DPR Wikipedia 索引脚本（FAISS）
├── verify_hotpotqa_setup.py           # 数据验证脚本
├── hotpotqa/                          # HotpotQA 数据集
│   ├── hotpot_dev_distractor_v1.json
│   └── hotpot_dev_fullwiki_v1.json
└── wiki/                              # Wikipedia 语料库
    ├── psgs_w100.tsv                  # DPR passages (21M, 13GB)
    └── enwiki-20171001-pages-meta-current-withlinks-abstracts/
```

---

## 🗄️ 向量数据库选择

本项目支持多种向量数据库后端，用于检索增强生成（RAG）：

| 数据库 | 运行模式 | 特点 | 适用场景 |
|--------|---------|------|---------|
| **Chroma** | CPU/GPU | 易用、轻量、持久化支持 | 开发测试、中小规模数据 |
| **FAISS** | CPU/GPU | 高性能、低延迟、内存索引 | 系统级基准测试、大规模数据 |

---

## 📦 1. 数据准备

### 1.1 下载 HotpotQA 数据集

#### 语料库(自己解压)
```bash
wget -c --tries=20 --waitretry=5 --timeout=30 \ -O hotpotqa_intro_paragraphs.tar.bz2 \ "https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2"
```

#### QA-distractor
```bash
curl -L -o hotpot_dev_distractor_v1.json \ "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
```

#### QA-fullwiki
```bash
wget -c --tries=20 --waitretry=5 --timeout=30 \
-O hotpot_dev_fullwiki_v1.json \
"http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json"
```

### 1.2 下载 Wiki papassages for DPR

#### 语料库
```bash
wget -c https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
```

#### 语料库解压
```bash
gunzip -k psgs_w100.tsv.gz
```

#### SQuAD v2.0
```bash
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
```

#### WebQuestions (clone)
```bash
git clone https://github.com/brmson/dataset-factoid-webquestions.git
```

#### TriviaQA
```bash
wget http://nlp.cs.washington.edu/triviaqa/data/triviaqa-v1.0.zip
wget http://nlp.cs.washington.edu/triviaqa/data/triviaqa-unfiltered-v1.0.zip
unzip triviaqa-v1.0.zip
unzip triviaqa-unfiltered-v1.0.zip
```

## 🔧 2. 向量数据库设置

### 方案 A: Chroma
#### 安装依赖

```bash
pip install langchain-chroma chromadb
```

#### 索引 HotpotQA FullWiki 数据

```bash
python index_hotpotqa_fullwiki.py \
    --wiki-dir data/wiki/enwiki-20171001-pages-meta-current-withlinks-abstracts \
    --persist-dir /mnt/Large_Language_Model_Lab_1/chroma_db/chroma_db_hotpotqa_fullwiki \
    --collection hotpotqa_fullwiki \
    --batch-size 500
```

**参数说明**：
- `--wiki-dir`: Wikipedia 语料库目录路径
- `--persist-dir`: Chroma 持久化目录（索引保存位置）
- `--collection`: 集合名称
- `--batch-size`: 批量插入大小（默认500，可根据内存调整）
- `--limit`: 限制处理的文章数量（默认0=全部，测试时可用 `--limit 1000`）


---

### 方案 B: FAISS（高性能）

FAISS（Facebook AI Similarity Search）是 Meta 开发的高性能向量检索库，特别适合系统级基准测试。


**CPU 版本**（适用于 CPU 检索）：
```bash
pip install faiss-cpu
```

**GPU 版本**（适用于 GPU 加速检索，推荐用于系统研究）：
```bash
# CUDA 11.x
pip install faiss-gpu

# CUDA 12.x (如果faiss-gpu不兼容)
conda install -c pytorch faiss-gpu
```

**验证安装**：
```bash
python -c "import faiss; print(faiss.__version__); print('GPU available:', hasattr(faiss, 'StandardGpuResources'))"
```

**安装 LangChain FAISS 适配器**

```bash
pip install langchain-community
```

### 索引 DPR Wikipedia Passages (推荐用于大规模基准测试)

`index_wiki_faiss.py` 专门用于索引 DPR 格式的 Wikipedia passages（`psgs_w100.tsv`，2100万条文档）。

#### 🎯 核心特性

1. **大规模数据优化**
   - ✅ **流式处理**：逐行读取 13GB TSV 文件，不会一次性加载到内存
   - ✅ **批量索引**：可配置批量大小（默认 10,000，GPU 模式建议 50,000）
   - ✅ **断点续传**：支持 `--resume` 参数，中断后可从检查点恢复
   - ✅ **进度显示**：使用 tqdm 显示实时进度和索引速度

2. **智能模型管理**
   - ✅ **统一存储目录**：`/mnt/Large_Language_Model_Lab_1/模型/rag_models`
   - ✅ **自动下载**：模型不存在时自动从 HuggingFace Hub 下载
   - ✅ **本地优先**：优先使用本地已有模型，避免重复下载
   - ✅ **缓存复用**：所有脚本共享同一模型缓存目录

3. **性能预估**
   - CPU 模式：~500-1000 docs/s
   - GPU 模式：~2000-5000 docs/s
   - 全量 2100 万条数据：GPU 模式约 1-2 小时

4. **检查点机制**
   ```
   faiss_index_wiki_dpr/
   ├── index.faiss       # FAISS 索引文件
   ├── index.pkl         # LangChain 元数据
   ├── checkpoint.json   # 检查点信息
   └── metadata.json     # 索引元数据
   ```
#### 使用示例

**全量索引（推荐，GPU 加速）：**
```bash
# 首次运行会自动下载模型到：
# /mnt/Large_Language_Model_Lab_1/模型/rag_models/
python data/index_wiki_faiss.py \
    --embedding-model /mnt/Large_Language_Model_Lab_1/模型/rag_models/BAAI-bge-base-en-v1.5 \
    --use-gpu \
    --index-dir /mnt/Large_Language_Model_Lab_1/faiss_index_wiki_dpr \
    --batch-size 50000 \
    --encode-batch-size 1024 \
    --resume

# 如果模型已缓存，会直接从本地加载
```

**使用自定义模型：**
```bash
# 使用其他 HuggingFace 模型
python data/index_wiki_faiss.py \
    --embedding-model intfloat/e5-large \
    --use-gpu \
    --index-dir /mnt/Large_Language_Model_Lab_1/faiss_index_wiki_dpr \
    --batch-size 100000 \
    --encode-batch-size 1024 \
    --resume


**断点续传：**
```bash
# 如果索引中断，使用 --resume 从检查点恢复
python data/index_wiki_faiss.py \
    --use-gpu \
    --resume
```

#### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--tsv-file` | `data/wiki/psgs_w100.tsv` | TSV 文件路径 |
| `--index-dir` | `/mnt/.../faiss_index_wiki_dpr` | 索引输出目录（持久化） |
| `--limit` | `0` (全部) | 限制文档数量（测试用） |
| `--batch-size` | `10000` | 批量大小（GPU建议50000） |
| `--encode-batch-size` | `1024` | Embedding 模型内部批量处理大小 |
| `--use-gpu` | `False` | GPU 加速（推荐开启） |
| `--embedding-model` | `BAAI/bge-base-en-v1.5` | HuggingFace 模型名或本地路径 |
| `--resume` | `False` | 断点续传（中断后恢复） |

**模型参数详解**：
- 默认使用 `BAAI/bge-base-en-v1.5`
- 自动管理：本地存在则加载，不存在则自动下载到 `/mnt/Large_Language_Model_Lab_1/模型/rag_models/`
- 支持任何 HuggingFace 上的 sentence-transformers 兼容模型


#### 验证索引

索引完成后验证：

```bash
python -c "
import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 设置模型缓存
model_base_dir = Path('/mnt/Large_Language_Model_Lab_1/模型/rag_models')
os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(model_base_dir)

# 加载模型（自动使用本地或下载）
embedding = HuggingFaceEmbeddings(
    model_name='BAAI/bge-base-en-v1.5',
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True},
)

# 加载索引
vs = FAISS.load_local(
    '/mnt/Large_Language_Model_Lab_1/faiss_index_wiki_dpr',
    embedding,
    allow_dangerous_deserialization=True
)

print(f'✓ Total documents: {vs.index.ntotal:,}')
results = vs.similarity_search('artificial intelligence', k=5)
print(f'✓ Sample result:')
print(f'  Title: {results[0].metadata.get(\"title\", \"N/A\")}')
print(f'  Text: {results[0].page_content[:200]}...')
"
```

---

## 🛠️ 3. 故障排查

### Chroma 常见问题

**问题**: `chromadb.errors.InvalidCollectionException: Collection not found`

**解决**:
```bash
# 检查集合是否存在
python -c "
import chromadb
client = chromadb.PersistentClient(path='./chroma_db_hotpotqa_fullwiki')
print(client.list_collections())
"

# 如果为空，需要重新索引
python index_hotpotqa_fullwiki.py ...
```

### FAISS 常见问题

**问题**: `RuntimeError: FAISS GPU not available`

**解决**:
```bash
# 检查 FAISS GPU 支持
python -c "import faiss; print(hasattr(faiss, 'StandardGpuResources'))"

# 如果返回 False，重新安装 GPU 版本
pip uninstall faiss-cpu faiss-gpu
conda install -c pytorch faiss-gpu
```

**问题**: `pickle.UnpicklingError` 或 `EOFError`

**解决**: FAISS 索引文件损坏，需要重新索引
```bash
rm -rf ./faiss_index_hotpotqa
python data/index_hotpotqa_faiss.py ...
```

### 嵌入模型问题

**问题 1**: 模型下载失败或超时

**解决**:
```bash
# 检查网络连接
curl -I https://huggingface.co

# 手动指定 HuggingFace 镜像（国内用户）
export HF_ENDPOINT=https://hf-mirror.com

# 或使用代理
export HTTP_PROXY=http://proxy:port
export HTTPS_PROXY=http://proxy:port
```

**问题 2**: 模型加载时维度不匹配

**解决**: 确保使用相同的模型创建和加载索引
```bash
# 检查索引使用的模型
cat /mnt/Large_Language_Model_Lab_1/faiss_index_wiki_dpr/metadata.json

# 使用相同模型加载
python data/index_wiki_faiss.py --embedding-model <same-model-name>
```

**问题 3**: 缓存目录权限不足

**解决**:
```bash
# 检查权限
ls -ld /mnt/Large_Language_Model_Lab_1/模型/rag_models

# 修改权限
sudo chmod -R 755 /mnt/Large_Language_Model_Lab_1/模型/rag_models
sudo chown -R $USER:$USER /mnt/Large_Language_Model_Lab_1/模型/rag_models
```

---


## 💡 最佳实践

1. **开发阶段**: 使用 Chroma + 小数据集（`--limit 10000`）快速迭代
2. **系统测试**: 使用 FAISS + 全量数据进行性能基准测试
3. **嵌入模型管理**:
   - 统一存储：所有模型放在 `/mnt/Large_Language_Model_Lab_1/模型/rag_models/`
   - 模型选择：
     - 小规模/开发: `BAAI/bge-base-en-v1.5` (768-dim, ~500MB)
     - 大规模/生产: `BAAI/bge-large-en-v1.5` (1024-dim, ~1.3GB)
   - 首次运行：使用小数据集测试模型下载是否正常
   - 模型复用：多个脚本共享同一模型缓存，避免重复下载
4. **批量大小**:
   - Chroma: 500-1000
   - FAISS CPU: 1000-2000
   - FAISS GPU: 5000-50000（根据 GPU 内存调整）
5. **断点续传**:
   - 全量索引时建议启用 `--resume`
   - 定期检查检查点文件（每10个batch自动保存）

---

**维护者**: 根据系统研究需求持续更新

**最后更新**: 2026-01-20
