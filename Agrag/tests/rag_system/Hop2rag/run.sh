conda activate langgraph_vllm

# 基础启动命令
python -m vllm.entrypoints.openai.api_server \
    --model /mnt/Large_Language_Model_Lab_1/模型/models/Qwen-Qwen2.5-7B-Instruct \
    --served-model-name Qwen2.5 \
    --dtype auto \
    --api-key EMPTY \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --port 8000

export AGRAG_PERSIST_DIR="/mnt/Large_Language_Model_Lab_1/chroma_db/chroma_db_hotpotqa_fullwiki"                                                
export AGRAG_COLLECTION_NAME="hotpot_fullwiki"   

python tests/rag_system/Hop2rag/test_hop2rag_latency.py \
        --limit 30 \
        --k 10 \
        --max-hops 3 \
        --enable-instrumentation