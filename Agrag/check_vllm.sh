echo "=== KV cache usage ==="
curl -s http://localhost:8000/metrics | grep vllm:kv_cache_usage_perc
echo " "
echo "=== Cache config ==="
curl -s http://localhost:8000/metrics | grep vllm:cache_config_info
echo " "
echo "=== Token counters ==="
curl -s http://localhost:8000/metrics | egrep "prompt_tokens_total|generation_tokens_total"
echo " "
echo "=== Requests ==="
curl -s http://localhost:8000/metrics | egrep "num_requests_running|num_requests_waiting"
echo " "
echo "=== vLLM process ==="
ps -ef | grep vllm | grep -v grep
echo " "
echo "=== GPU ==="
nvidia-smi


#vllm
uv venv --python 3.12 --seed
source .venv/bin/activate