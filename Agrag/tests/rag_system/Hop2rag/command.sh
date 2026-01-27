conda activate langgraph_vllm

# 路径显示输出
export AGRAG_PERSIST_DIR="/mnt/Large_Language_Model_Lab_1/chroma_db/chroma_db_hotpotqa_fullwiki"                                                
export AGRAG_COLLECTION_NAME="hotpotqa_fullwiki"   

# 基础启动命令
nsys profile -o ../report.nsys-rep \
    --trace=cuda,nvtx \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
    --nvtx-capture=layerwise \
    --force-overwrite true \
python -m vllm.entrypoints.openai.api_server \
    --model /mnt/Large_Language_Model_Lab_1/模型/models/Qwen-Qwen2.5-7B-Instruct \
    --served-model-name Qwen2.5 \
    --enable-prefix-caching \
    --enable-layerwise-nvtx-tracing \
    --disable-log-requests \
    --dtype auto \
    --api-key EMPTY \
    --port 8000

# 纯净版hop2rag测试
python tests/rag_system/Hop2rag/test_hop2rag_performance.py \
            --limit 10 \
            --k 10 \
            --monitor-interval 0.5 \
            --max-hops 10

# 结束 profiling（关键）
nsys sessions list
nsys stop --session=profile-<>

# 使用nsys查看报告
nsys stats --help
nsys stats report.nsys-rep
# kernel 级时间线
nsys stats report.nsys-rep --report cuda_gpu_trace --format csv,column --output  Agrag/tests/results/hop2rag_performance/cuda_gpu_trace.csv
# OSRT 级时间线
nsys stats report.nsys-rep --report osrt_trace --format csv,column --output  Agrag/tests/results/hop2rag_performance/osrt_trace.csv
# vLLM / NVTX 区间
nsys stats report.nsys-rep --report nvtx_sum --format csv,column --output  Agrag/tests/results/hop2rag_performance/nvtx_sum.csv
# cudaMalloc / sync 统计
nsys stats report.nsys-rep --report cuda_api_sum --format csv,column --output  Agrag/tests/results/hop2rag_performance/api_summary.csv
# CUDA 内存使用情况
nsys stats report.nsys-rep --report cuda_memory_usage --format csv,column --output  Agrag/tests/results/hop2rag_performance/cuda_memory_usage.csv

# 画图
python tests/rag_system/Hop2rag/plot_gpu_execution.py 

# 检查vllm性能指标是否正确
curl -s http://localhost:8000/metrics | egrep "kv_cache|gpu_cache|cache_usage" | head

# 检查库是否有数据
python - << 'PY'
import os, chromadb
persist = os.environ["AGRAG_PERSIST_DIR"]
client = chromadb.PersistentClient(path=persist)

for name in ["hotpot_fullwiki", "hotpotqa_fullwiki", "rag-chroma"]:
    try:
        col = client.get_collection(name)
        print(name, "count=", col.count())
    except Exception as e:
        print(name, "ERROR", e)
PY

