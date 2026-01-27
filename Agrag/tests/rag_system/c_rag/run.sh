conda activate rag_adaptive

# 基础启动命令
python -m vllm.entrypoints.openai.api_server \
    --model /mnt/Large_Language_Model_Lab_1/模型/models/Qwen-Qwen2.5-7B-Instruct \
    --served-model-name Qwen2.5 \
    --enable-prefix-caching \
    --disable-log-requests \
    --dtype auto \
    --api-key EMPTY \
    --port 8000

export AGRAG_FAISS_DIR="/mnt/Large_Language_Model_Lab_1/faiss_wiki_db"      

#纯净版测试
python tests/rag_system/c_rag/test_c_rag_performance.py \
        --limit 10 \
        --monitor-interval 0.5 \

python tests/rag_system/c_rag/test_crag_on_squad_dev.py \
  --squad-dev tests/rag_system/c_rag/data/SQUAD-dev-v2.0.json \
  --start 100 \
  --limit 10 \
  --monitor-interval 0.5 \
  --verbose
