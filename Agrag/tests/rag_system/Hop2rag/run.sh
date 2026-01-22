conda activate langgraph_vllm

# 基础启动命令
python -m vllm.entrypoints.openai.api_server \
    --model /mnt/Large_Language_Model_Lab_1/模型/models/Qwen-Qwen2.5-7B-Instruct \
    --served-model-name Qwen2.5 \
    --enable-prefix-caching \
    --disable-log-requests \
    --dtype auto \
    --api-key EMPTY \
    --port 8000

# 路径显示输出
export AGRAG_PERSIST_DIR="/mnt/Large_Language_Model_Lab_1/chroma_db/chroma_db_hotpotqa_fullwiki"                                                
export AGRAG_COLLECTION_NAME="hotpotqa_fullwiki"   

python tests/rag_system/Hop2rag/test_hop2rag_latency.py \
        --limit 50 \
        --k 20 \
        --max-hops 5 \
        --enable-instrumentation

# 纯净版hop2rag测试
python tests/rag_system/Hop2rag/test_hop2rag_performance.py \
        --limit 10 \
        --k 10 \
        --monitor-interval 0.5 \
        --max-hops 5
        
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


#实验一：KV Cache 复用潜力分析
python tests/rag_system/Hop2rag/test_kv_cache_analysis.py  


# 实验二：GPU 内存时序分析
# vLLM 配置
python -m vllm.entrypoints.openai.api_server \
       --model /mnt/Large_Language_Model_Lab_1/模型/models/Qwen-Qwen2.5-7B-Instruct \
       --served-model-name Qwen2.5 \
       --port 8000 \
       --disable-log-requests

python tests/rag_system/Hop2rag/test_gpu_memory_profile.py --limit 10 --k 20 --max-hops 10 --sample-interval 10