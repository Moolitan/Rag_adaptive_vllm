from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="/mnt/Large_Language_Model_Lab_1/模型/rag_models/BAAI-bge-base-en-v1.5",
    encode_kwargs={'normalize_embeddings': True},
)

db = Chroma(
    persist_directory="/mnt/Large_Language_Model_Lab_1/chroma_db/chroma_db_hotpotqa_fullwiki",
    embedding_function=embedding,
    collection_name="hotpotqa_fullwiki",
)

print("文档数量:", db._collection.count())
