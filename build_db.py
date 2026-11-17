"""独立建库脚本：读 docs → 按二级标题切块 → bge-m3 向量化 → 重建 chroma_db。

跑法：python build_db.py（在 rag/ 目录下；从别的工作目录跑也能定位路径）
改了 docs/ 后重跑本脚本即可：先删旧 collection 再整体重建，不残留旧块。
retrieval.py 只负责检索，两个脚本共用同一个 chroma_db。
"""

import os
from pathlib import Path

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent

load_dotenv()

embed_client = OpenAI(
    base_url="https://api.siliconflow.cn/v1",
    api_key=os.environ["SILICONFLOW_API_KEY"],
)
EMBED_MODEL = "BAAI/bge-m3"

# 1.拿到 docs 下全部 md，供切块遍历
doc_files = sorted((BASE_DIR / "docs").glob("*.md"))


def split_chunks():
    """读 docs 并按二级标题分块（与历史 retrieval 切块逻辑一致）"""
    chunks = []
    for i, path in enumerate(doc_files, start=1):
        text = path.read_text(encoding="utf-8")
        parts = text.split("\n## ")
        for j, chunk in enumerate(parts):
            if j == 0:
                continue
            body = "## " + chunk.strip()
            if j == 1:  # 第一小节：把一级标题拼到最前面
                body = parts[0].strip() + "\n\n" + body
            chunks.append({"文件序号": i, "文件名": path.name, "文件内容": body})
    return chunks


def embed(texts):
    """批量文本 → bge-m3 向量，返回顺序与输入一一对应"""
    resp = embed_client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]


chunks = split_chunks()
print(f"切块完成  : {len(chunks)} 个 chunk")

# 2.先向量化（调 API）：成功后再动旧库，避免删库后建库失败留下空库
vectors = embed([c["文件内容"] for c in chunks])
print(f"向量化完成: {len(vectors)} 个向量\n")

# 3.连接 chroma，删掉旧 collection 再整体重建，保证旧块不残留
chroma_client = chromadb.PersistentClient(
    path=str(BASE_DIR / "chroma_db"),
    settings=Settings(anonymized_telemetry=False),
)
if "hr_docs" in [c.name for c in chroma_client.list_collections()]:
    chroma_client.delete_collection("hr_docs")
    print("已删除旧 collection: hr_docs")

collection = chroma_client.get_or_create_collection(
    name="hr_docs",
    metadata={"hnsw:space": "cosine"},
)

# 4.本地写入（向量 + 原文 + 文件名 metadata），不再依赖网络
collection.add(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    embeddings=vectors,
    documents=[c["文件内容"] for c in chunks],
    metadatas=[{"文件名": c["文件名"]} for c in chunks],
)
print(f"建库完成  : {collection.count()} 个 chunk 已写入 {BASE_DIR / 'chroma_db'}")
