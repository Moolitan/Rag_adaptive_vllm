conda activate rag_adaptive
#vllm启动
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
export TAVILY_API_KEY="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"
#此处目录位置是Agrag
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
#push到github上
git push -u origin self_rag性能测试分析
#推送不了时，使用这个命令
git remote set-url origin https://github.com/Moolitan/Rag_adaptive_vllm.git

# 前台运行faiss数据库服务,注意目录位置
cd /home/wjj/Rag_adaptive_vllm
AGRAG_FAISS_DIR="/mnt/Large_Language_Model_Lab_1/faiss_wiki_db" python -m Agrag.Rag.faiss_server
#或选择后台运行faiss数据库服务
nohup env AGRAG_FAISS_DIR="/mnt/Large_Language_Model_Lab_1/faiss_wiki_db" -m Agrag.Rag.faiss_server > faiss_server.log 2>&1 &
#检查faiss数据库服务状态
curl http://127.0.0.1:5100/health