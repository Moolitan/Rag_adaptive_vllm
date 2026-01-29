"""
FAISS Retriever 客户端
用于连接后台运行的 faiss_server.py

使用方式：
    from Agrag.Rag.faiss_client import RemoteRetriever
    
    retriever = RemoteRetriever()  # 连接到 localhost:5100
    docs = retriever.invoke("What is machine learning?")
"""
import requests
from typing import List, Optional
from langchain_core.documents import Document


class RemoteRetriever:
    """远程 FAISS Retriever 客户端"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 5100, timeout: int = 30):
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.search_kwargs = {"k": 15}
    
    def is_available(self) -> bool:
        """检查服务是否可用"""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200 and resp.json().get("ready", False)
        except:
            return False
    
    def invoke(self, query: str) -> List[Document]:
        """执行检索"""
        k = self.search_kwargs.get("k", 15)
        
        resp = requests.post(
            f"{self.base_url}/retrieve",
            json={"query": query, "k": k},
            timeout=self.timeout
        )
        resp.raise_for_status()
        
        data = resp.json()
        docs = [
            Document(
                page_content=d["page_content"],
                metadata=d.get("metadata", {})
            )
            for d in data.get("documents", [])
        ]
        return docs
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """兼容 LangChain retriever 接口"""
        return self.invoke(query)


def get_remote_retriever(host: str = "127.0.0.1", port: int = 5100) -> RemoteRetriever:
    """获取远程 retriever 实例"""
    return RemoteRetriever(host=host, port=port)
