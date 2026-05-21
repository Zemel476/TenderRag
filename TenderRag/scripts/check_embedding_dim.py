"""
验证 embedding 维度是否与 Chroma 索引匹配。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from app.config import settings
from llama_index.embeddings.dashscope import DashScopeEmbedding

# 1. 查看当前 embedding 模型的向量维度
embed_model = DashScopeEmbedding(
    model_name=settings.dashscope_embedding_model,
    api_key=settings.dashscope_api_key,
)

test_embedding = embed_model.get_text_embedding("测试")
print(f"Embedding model: {settings.dashscope_embedding_model}")
print(f"Embedding dimension: {len(test_embedding)}")

# 2. 查看 Chroma 集合中存储的向量维度
chroma_client = chromadb.PersistentClient(path=settings.chroma_db_path)

for name in ["legal_db", "tender_db"]:
    try:
        collection = chroma_client.get_collection(name)
        result = collection.peek(limit=1)
        if result["embeddings"] and len(result["embeddings"]) > 0:
            dim = len(result["embeddings"][0])
            print(f"{name} embedding dimension: {dim}")
        else:
            print(f"{name} is empty.")
    except Exception as e:
        print(f"{name} check failed: {e}")
