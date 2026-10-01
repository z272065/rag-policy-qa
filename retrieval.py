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


# 1.定位 docs 目录
BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"

if not DOCS_DIR.is_dir():
    raise SystemExit(f"找不到目录: {DOCS_DIR}（当前工作目录是 {Path.cwd()}）")

doc_files = sorted(DOCS_DIR.glob("*.md"))

print(f"docs 目录 : {DOCS_DIR}")
print(f"找到文件  : {len(doc_files)} 个\n")


def split_chunks():
    """读 docs 并按二级标题分块"""
    chunks = []
    for i, path in enumerate(doc_files, start=1):
        text = path.read_text(encoding="utf-8")
        parts = text.split("\n## ")
        for j, chunk in enumerate(parts):
            if j == 0:
                continue
            body = "## " + chunk.strip()
            if j == 1:
                body = parts[0].strip() + "\n\n" + body
            chunks.append({"文件序号": i, "文件名": path.name, "文件内容": body})
    return chunks


def embed(texts):
    """把一批文本变成向量（每段文本 -> 1024 个浮点数）。

    返回值的顺序和 input 的顺序一一对应：第 i 个向量就是第 i 段文本的。
    """
    resp = embed_client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]


chunks = split_chunks()

# 2.建向量库：第一次运行才 embedding，之后直接读盘（向量是内容的函数，内容不变就不用重算）
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

if collection.count() == 0:
    # 库是空的（第一次运行）：调 API 把全部 chunk 向量化，写入库
    vectors = embed([c["文件内容"] for c in chunks])
    collection.add(
        ids=[
            f"chunk_{i}" for i in range(len(chunks))
        ],  # chroma 要求每条有唯一字符串 id
        embeddings=vectors,  # 向量本体（1024 维 float）
        documents=[c["文件内容"] for c in chunks],  # 原文也存，查到后直接拿回文本
        metadatas=[{"文件名": c["文件名"]} for c in chunks],
    )
    print(f"首次运行，已向量化入库: {collection.count()} 个 chunk\n")
else:
    print(f"复用已有向量库: {collection.count()} 个 chunk（本次没调 embedding API）\n")
    # 注意：改了 docs/ 里的文档向量库不会自动更新，删掉 rag/chroma_db/ 再运行即可重建。


def search_knowledge_base(user_prompt):
    # 3.向量检索：问题也用同一个 bge-m3 编码（同一把尺子，距离才有意义；换模型要删库重建）
    query_vec = embed([user_prompt])[0]

    result = collection.query(query_embeddings=[query_vec], n_results=3)

    # result 每层都是列表套列表（query 支持一次多问），[0] 是第 0 个问题的结果
    for rank, (doc, meta, dist) in enumerate(
        zip(result["documents"][0], result["metadatas"][0], result["distances"][0]),
        start=1,
    ):
        preview = doc[:30].replace("\n", " ")
        print(
            f"第{rank}名, 相似度:{1 - dist:.3f}, 文件名:{meta['文件名']}, 内容开头:{preview}"
        )

    top_content = ""
    for doc, meta in zip(result["documents"][0], result["metadatas"][0]):
        top_content += f"【{meta['文件名']}】\n{doc}\n\n"
    return top_content


if __name__ == "__main__":
    print(search_knowledge_base("我想休假，要提前几天申请？"))
