"""20 题评测：关键词版 vs 向量版。

跑法（在 rag/ 目录下）：python experiments/eval_run.py

1. 检索层对比：20 题分别用双字打分和向量检索给 15 个块排名，
   看答案块排第几、进没进 Top-3（关键词零 API，向量每题 1 次 embedding），
   名次存 eval20_ranks.json
2. 生成层（只向量版）：Top-3 拼材料调 deepseek-flash，20 条回答存
   eval20_results_raw.json（机器初稿，可覆盖）；人工打完标签补进
   eval20_results.json（证据文件，脚本永不覆盖）。关键词版生成沿用已有 5 题数据，不重复跑

注意：docs/ 内容改动后要删掉 rag/chroma_db/ 再跑，否则用的是旧向量库。
"""

import json
import os
from pathlib import Path

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

chat_client = OpenAI(
    base_url="https://api.deepseek.com", api_key=os.environ["DEEPSEEK_API_KEY"]
)
embed_client = OpenAI(
    base_url="https://api.siliconflow.cn/v1", api_key=os.environ["SILICONFLOW_API_KEY"]
)
EMBED_MODEL = "BAAI/bge-m3"

BASE_DIR = Path(__file__).resolve().parent.parent  # rag/
DOCS_DIR = BASE_DIR / "docs"
EVAL_FILE = Path(__file__).resolve().parent / "eval20.json"
RANKS_FILE = Path(__file__).resolve().parent / "eval20_ranks.json"
RESULTS_FILE = Path(__file__).resolve().parent / "eval20_results_raw.json"

SYSTEM_PROMPT = "你是一个hr助手,需要根据材料如实回答,不清楚不确定的问题绝对不能编"


def build_chunks():
    """切块：与 rag_demo.py / rag_demo_vector.py 完全相同。"""
    chunks = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        parts = text.split("\n## ")
        for j, chunk in enumerate(parts):
            if j == 0:
                continue
            body = "## " + chunk.strip()
            if j == 1:
                body = parts[0].strip() + "\n\n" + body
            chunks.append({"文件名": path.name, "文件内容": body})
    return chunks


def kw_score(question, text):
    """关键词版打分：问题里每个相邻双字，在块里出现就 +1。"""
    return sum(1 for k in range(len(question) - 1) if question[k : k + 2] in text)


def kw_rank(question, chunks):
    """关键词版全量排名。sorted 是稳定排序，同分保持 chunks 原顺序——和 rag_demo 一致。"""
    return sorted(chunks, key=lambda c: kw_score(question, c["文件内容"]), reverse=True)


def embed(texts):
    resp = embed_client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]


def get_collection(chunks):
    """打开（或首次创建）向量库，逻辑同 rag_demo_vector.py。"""
    client = chromadb.PersistentClient(
        path=str(BASE_DIR / "chroma_db"),
        settings=Settings(anonymized_telemetry=False),
    )
    coll = client.get_or_create_collection(
        name="hr_docs", metadata={"hnsw:space": "cosine"}
    )
    if coll.count() == 0:
        vectors = embed([c["文件内容"] for c in chunks])
        coll.add(
            ids=[f"chunk_{i}" for i in range(len(chunks))],
            embeddings=vectors,
            documents=[c["文件内容"] for c in chunks],
            metadatas=[{"文件名": c["文件名"]} for c in chunks],
        )
        print(f"首次建库：{coll.count()} 个块已向量化\n")
    return coll


def vec_rank(question, coll, n):
    """向量版全量排名，返回 [(文档文本, 文件名), ...] 按相似度从高到低。"""
    qv = embed([question])[0]
    res = coll.query(query_embeddings=[qv], n_results=n)
    return list(zip(res["documents"][0], [m["文件名"] for m in res["metadatas"][0]]))


def is_answer_block(text, filename, answer_block):
    """这块是不是这道题的答案块：文件名对上，且块里有对应小节标题。"""
    return filename == answer_block["文件名"] and f"## {answer_block['小节']}" in text


def main():
    questions = json.loads(EVAL_FILE.read_text(encoding="utf-8"))["题目"]
    chunks = build_chunks()
    coll = get_collection(chunks)
    n = len(chunks)

    # ---- 1) 检索层对比 ------------------------------------------------
    print(f"{'编号':<4}{'类别':<6}{'关键词排名':<10}{'向量排名':<8}问题")
    summary = {
        "直球": [0, 0],
        "换说法": [0, 0],
        "库外": [0, 0],
    }  # [关键词进Top3次数, 向量进Top3次数]
    ranks = []  # 每题的检索名次，循环里逐题追加，检索循环结束后一次性落盘
    for q in questions:
        kw_pos = vec_pos = None
        if q["答案块"]:
            ranked_kw = kw_rank(q["问题"], chunks)
            kw_pos = next(
                (
                    i + 1
                    for i, c in enumerate(ranked_kw)
                    if is_answer_block(c["文件内容"], c["文件名"], q["答案块"])
                ),
                None,
            )
            ranked_vec = vec_rank(q["问题"], coll, n)
            vec_pos = next(
                (
                    i + 1
                    for i, (doc, fname) in enumerate(ranked_vec)
                    if is_answer_block(doc, fname, q["答案块"])
                ),
                None,
            )
            if kw_pos and kw_pos <= 3:
                summary[q["类别"]][0] += 1
            if vec_pos and vec_pos <= 3:
                summary[q["类别"]][1] += 1
        ranks.append(
            {
                "编号": q["编号"],
                "类别": q["类别"],
                "关键词名次": kw_pos,
                "向量名次": vec_pos,
            }
        )
        print(
            f"{q['编号']:<4}{q['类别']:<6}{str(kw_pos or '-'):<10}{str(vec_pos or '-'):<8}{q['问题'][:24]}"
        )

    print("\n按类别统计 Top-3 命中（库外题没有答案块，不参与）：")
    for cat, (k, v) in summary.items():
        print(f"  {cat}: 关键词 {k} 次 / 向量 {v} 次")

    # 分段落盘：检索层先落档，生成层再开跑——后面挂了排名证据也已经在盘上
    RANKS_FILE.write_text(
        json.dumps(ranks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n检索名次已存 {RANKS_FILE}")

    # ---- 2) 生成层（向量版） -------------------------------------------
    print("\n开始生成向量版回答（20 次对话调用）...")
    results = []
    for q in questions:
        top3 = vec_rank(q["问题"], coll, 3)
        material = "".join(f"【{fname}】\n{doc}\n\n" for doc, fname in top3)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{material}\n{q['问题']}"},
        ]
        resp = chat_client.chat.completions.create(
            model="deepseek-flash", messages=messages
        )
        results.append(
            {
                "编号": q["编号"],
                "类别": q["类别"],
                "问题": q["问题"],
                "标准答案": q["标准答案"],
                "Top3": [fname for _, fname in top3],
                "回答": resp.choices[0].message.content,
            }
        )
        print(f"  第 {q['编号']} 题完成")

    RESULTS_FILE.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"\n已存 {RESULTS_FILE}（机器初稿）。逐条读完把「人工判定」补进"
        f" eval20_results.json——那是证据文件，脚本不会碰它。"
    )


if __name__ == "__main__":
    main()
