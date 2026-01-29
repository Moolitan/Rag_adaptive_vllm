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

# - bridge：桥接型问题（5918个） - 需要通过中间实体连接到最终答案                                 
# - 例如："Who is the director of the movie that won Best Picture at the 2020 Oscars?"          
# - 需要先找到"哪部电影获奖" → 再找"该电影的导演是谁"                                           
# - comparison：比较型问题（1487个） - 需要比较两个实体的某个属性                                 
# - 例如："Were Scott Derrickson and Ed Wood of the same nationality?"                          
# - 需要分别查找两个人的国籍，然后比较 
# 纯净版hop2rag测试（并发版）
python tests/rag_system/Hop2rag/test_hop2rag_performance_concurrent.py \
    --limit 100 \
    --k 10 \
    --max-workers 250 \
    --question-type bridge \
    --monitor-interval 0.5  \
    --level hard \
    --max-hops 10  

# 结束 profiling（关键）
nsys sessions list
nsys stop --session=profile-<>

# 使用nsys查看报告
nsys stats --help
nsys stats report.nsys-rep


# 分析 Prompt Token 分布并获取建议的 Threshold
python tests/rag_system/Hop2rag/analyze_prompt_distribution.py \
    --input tests/results/hop2rag_performance/llm_calls.json \
    --output tests/results/hop2rag_performance/plots/prompt_dist/prompt_dist \


# 绘制自己写的Vllm监控器的前缀命中率的图表                                                                                        
python tests/rag_system/Hop2rag/plot_prefix_cache_hitrate.py \
    --input tests/results/hop2rag_performance/vllm_metrics.csv \
    --output tests/results/hop2rag_performance/plots/prefix_cache_hitrate.png 


# 画并发测试的性能指标图
python tests/rag_system/Hop2rag/plot_concurrent_metrics.py \
    --results-dir tests/results/hop2rag_performance_concurrent \
    --limit 100 \
    --output tests/results/hop2rag_performance_concurrent/plots 

# 画nsys profile采样的图
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


