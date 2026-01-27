
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
    --delay=33 \
    --duration=200 \
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
