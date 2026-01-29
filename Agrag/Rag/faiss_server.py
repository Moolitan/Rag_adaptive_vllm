"""
FAISS Retriever 独立服务端
启动后会一直在后台运行，测试程序可以通过 HTTP 调用

使用方式：
    启动服务: python faiss_server.py
    或后台运行: nohup python faiss_server.py > faiss_server.log 2>&1 &
"""
import os
import json
from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

app = Flask(__name__)

# 全局 retriever 实例
_retriever = None
_vectorstore = None


def init_faiss():
    """初始化 FAISS 数据库"""
    global _retriever, _vectorstore
    
    if _retriever is not None:
        return
    
    faiss_dir = os.environ.get("AGRAG_FAISS_DIR")
    if not faiss_dir:
        raise RuntimeError("Please set AGRAG_FAISS_DIR environment variable")
    
    print("Loading embedding model...")
    embedding = HuggingFaceEmbeddings(
        model_name="/mnt/Large_Language_Model_Lab_1/模型/rag_models/BAAI-bge-base-en-v1.5",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )
    
    print(f"Loading FAISS index from {faiss_dir}...")
    _vectorstore = FAISS.load_local(
        faiss_dir,
        embedding,
        allow_dangerous_deserialization=True,
    )
    
    _retriever = _vectorstore.as_retriever(search_kwargs={"k": 15})
    print("FAISS server initialized and ready!")


@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({"status": "ok", "ready": _retriever is not None})


@app.route('/retrieve', methods=['POST'])
def retrieve():
    """检索接口"""
    if _retriever is None:
        return jsonify({"error": "Retriever not initialized"}), 503
    
    data = request.get_json()
    query = data.get("query", "")
    k = data.get("k", 15)
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    # 动态调整 k 值
    _retriever.search_kwargs["k"] = k
    
    docs = _retriever.invoke(query)
    
    # 序列化文档
    results = [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in docs
    ]
    
    return jsonify({"documents": results, "count": len(results)})


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FAISS Retriever Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5100, help="Port to bind (default: 5100)")
    args = parser.parse_args()
    
    # 启动时初始化
    init_faiss()
    
    print(f"Starting FAISS server on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)
