conda activate langgraph_vllm

python -m vllm.entrypoints.openai.api_server \
    --model /mnt/Large_Language_Model_Lab_1/模型/models/Qwen-Qwen2.5-7B-Instruct \
    --served-model-name Qwen2.5 \
    --enable-prefix-caching \
    --enable-layerwise-nvtx-tracing \
    --disable-log-requests \
    --dtype auto \
    --api-key EMPTY \
    --port 8000

python tests/vllm/test_vllm_kv_cache.py \
    --iterations 30 \
    --max-tokens 20480 \
    --batch-size 10 \
    --monitor-interval 0.5 \
    --output-dir tests/results/vllm_kv_cache/


python tests/vllm/plot_kv_cache_metrics.py \
    --csv tests/results/vllm_kv_cache/vllm_metrics.csv \
    --output tests/results/vllm_kv_cache/plots  
