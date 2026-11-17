import os
import chromadb
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from chromadb.config import Settings

load_dotenv()

embed_client = OpenAI(
    base_url="https://api.siliconflow.cn/v1",
    api_key=os.environ["SILICONFLOW_API_KEY"],
)
EMBED_MODEL = "BAAI/bge-m3"


# 1.项目根目录（用于定位已有的 chroma_db 向量库）
BASE_DIR = Path(__file__).resolve().parent


def embed(texts):
    """把一批文本变成向量（每段文本 -> 1024 个浮点数）。

    返回值的顺序和 input 的顺序一一对应：第 i 个向量就是第 i 段文本的。
    """
    resp = embed_client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]


# 2.连接已有向量库（不再从 docs 自动建库，库由外部单独维护）
# anonymized_telemetry=False：关掉 chroma 的匿名统计
chroma_client = chromadb.PersistentClient(
    path=str(BASE_DIR / "chroma_db"),
    settings=Settings(anonymized_telemetry=False),
)

# metadata 里指定 cosine：余弦距离 = 1 - 余弦相似度，比"方向"（意思）不比"长度"（篇幅）
collection = chroma_client.get_or_create_collection(
    name="hr_docs",
    metadata={"hnsw:space": "cosine"},
)
print(f"连接向量库: {collection.count()} 个 chunk\n")


def search_knowledge_base(user_prompt):
    # 3.向量检索：问题也用同一个 bge-m3 编码（同一把尺子，距离才有意义；换模型要删库重建）
    query_vec = embed([user_prompt])[0]

    result = collection.query(query_embeddings=[query_vec], n_results=3)
    evidence = []
    # result 每层都是列表套列表（query 支持一次多问），[0] 是第 0 个问题的结果
    for rank, (doc, meta, dist) in enumerate(
        zip(result["documents"][0], result["metadatas"][0], result["distances"][0]),
        start=1,
    ):
        preview = doc[:30].replace("\n", " ")
        print(
            f"第{rank}名, 相似度:{1 - dist:.3f}, 文件名:{meta['文件名']}, 内容开头:{preview}"
        )
        evidence.append({"相似度": 1 - dist, "文件名": meta["文件名"], "文件内容": doc})
    return evidence


if __name__ == "__main__":
    print(search_knowledge_base("我想休假，要提前几天申请？"))
