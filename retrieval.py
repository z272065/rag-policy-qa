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


def kw_score(question, text):
    """关键词路打分：问题里每个相邻双字在块中出现就 +1"""
    return sum(1 for k in range(len(question) - 1) if question[k : k + 2] in text)


def hybrid_search_knowledge_base(user_prompt, candidate_size=20, n_results=3, rrf_constant=60):
    """混合检索：关键词双字 + 向量余弦两路，RRF 融合排名。

    流程：collection.get 拿全部块 → 关键词路打分取候选、向量路 query 取候选
    → 按 RRF 公式 1/(rrf_constant+名次) 累加两路名次 → 按总分降序取 Top3。
    只在某一路入候选的块，另一路贡献为 0、不参与那一项。
    """
    # 1.全量块：get 默认返回 ids/documents/metadatas（不含向量），按 chroma id 索引
    all_records = collection.get()
    block_by_id = {
        chunk_id: {"文件名": meta["文件名"], "文件内容": doc}
        for chunk_id, doc, meta in zip(
            all_records["ids"], all_records["documents"], all_records["metadatas"]
        )
    }

    # 2a.关键词路：双字打分、过滤零分、降序取候选
    kw_ranked = sorted(
        (
            (chunk_id, kw_score(user_prompt, block["文件内容"]))
            for chunk_id, block in block_by_id.items()
        ),
        key=lambda x: x[1],
        reverse=True,
    )
    kw_top = [
        (chunk_id, score) for chunk_id, score in kw_ranked if score > 0
    ][:candidate_size]

    # 2b.向量路：问题向量化后 query 候选，query 结果自带 ids
    query_vec = embed([user_prompt])[0]
    vector_results = collection.query(
        query_embeddings=[query_vec], n_results=candidate_size
    )

    # 3.RRF 字典累加：键是 chroma id，值存总分和两路名次
    rrf = {}
    for rank, (chunk_id, _) in enumerate(kw_top, start=1):
        rrf_entry = rrf.setdefault(
            chunk_id, {"rrf": 0.0, "关键词名次": None, "向量名次": None}
        )
        rrf_entry["rrf"] += 1 / (rrf_constant + rank)
        rrf_entry["关键词名次"] = rank

    for rank, chunk_id in enumerate(vector_results["ids"][0], start=1):
        rrf_entry = rrf.setdefault(
            chunk_id, {"rrf": 0.0, "关键词名次": None, "向量名次": None}
        )
        rrf_entry["rrf"] += 1 / (rrf_constant + rank)
        rrf_entry["向量名次"] = rank

    # 4.按 RRF 总分降序取 Top3，组装 evidence
    ranked_ids = sorted(rrf, key=lambda x: rrf[x]["rrf"], reverse=True)[:n_results]
    evidence = []
    for final_rank, chunk_id in enumerate(ranked_ids, start=1):
        block = block_by_id[chunk_id]
        rrf_entry = rrf[chunk_id]
        print(
            f"混合第{final_rank}名,RRF分:{rrf_entry['rrf']:.5f},"
            f"关键词名次:{rrf_entry['关键词名次']},向量名次:{rrf_entry['向量名次']},"
            f"文件名:{block['文件名']}"
        )
        evidence.append(
            {
                "混合名次": final_rank,
                "RRF分": rrf_entry["rrf"],
                "关键词名次": rrf_entry["关键词名次"],
                "向量名次": rrf_entry["向量名次"],
                "文件名": block["文件名"],
                "文件内容": block["文件内容"],
            }
        )
    return evidence


if __name__ == "__main__":
    print(hybrid_search_knowledge_base("我想休假，要提前几天申请？"))
