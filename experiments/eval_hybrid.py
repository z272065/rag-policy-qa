# -*- coding: utf-8 -*-
"""三路名次评测：同一题的答案块在关键词/向量/混合三路分别排第几。

环境与数据对象
- 复用 retrieval.py 的现役 collection 与函数，不自己切块、不自己建库
- 考卷：同目录 eval20.json（前 15 题有答案块，后 5 题库外不参与检索名次）
- 数据：当前 chroma_db（209 块，由 build_db.py 生成）
- API：每题向量路 1 次问题 embedding + 混合路内部 1 次，关键词路零 API

跑法（在 rag 目录）
    python experiments/eval_hybrid.py
结果落 eval_hybrid_ranks.json
"""

import json
import sys
from pathlib import Path

# 脚本在 rag/experiments 下，把父目录 rag 加入模块搜索路径才能 import retrieval
RAG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAG_DIR))

from retrieval import (  # noqa: E402
    collection,
    embed,
    hybrid_search_knowledge_base,
    kw_score,
)

EXPERIMENTS_DIR = Path(__file__).resolve().parent
EVAL_FILE = EXPERIMENTS_DIR / "eval20.json"
RANKS_FILE = EXPERIMENTS_DIR / "eval_hybrid_ranks.json"

TOTAL = collection.count()


def locate_answer_id(answer_block, ids, id_to_doc, id_to_file):
    """按文件名+小节标题定位答案块的 chroma id；找不到返回 None。"""
    return next(
        (
            chunk_id
            for chunk_id in ids
            if id_to_file[chunk_id] == answer_block["文件名"]
            and f"## {answer_block['小节']}" in id_to_doc[chunk_id]
        ),
        None,
    )


def main():
    questions = json.loads(EVAL_FILE.read_text(encoding="utf-8"))["题目"]

    # 全量块只 get 一次，三路判定共用
    records = collection.get()
    ids = records["ids"]
    id_to_doc = dict(zip(ids, records["documents"]))
    id_to_file = {
        chunk_id: meta["文件名"]
        for chunk_id, meta in zip(ids, records["metadatas"])
    }

    print(f"{'编号':<4}{'类别':<6}{'关键词':<7}{'向量':<7}{'混合':<7}问题")
    rows = []
    for q in questions:
        answer_block = q["答案块"]
        if answer_block is None:
            # 库外题没有答案块，名次留空
            rows.append(
                {
                    "编号": q["编号"],
                    "类别": q["类别"],
                    "问题": q["问题"],
                    "关键词名次": None,
                    "向量名次": None,
                    "混合名次": None,
                }
            )
            print(
                f"{q['编号']:<4}{q['类别']:<6}{'-':<7}{'-':<7}{'-':<7}{q['问题'][:20]}"
            )
            continue

        answer_id = locate_answer_id(answer_block, ids, id_to_doc, id_to_file)

        # 1.关键词路：双字打分降序，找答案块名次
        kw_sorted = sorted(
            ((chunk_id, kw_score(q["问题"], id_to_doc[chunk_id])) for chunk_id in ids),
            key=lambda x: x[1],
            reverse=True,
        )
        kw_pos = next(
            (i + 1 for i, (chunk_id, _) in enumerate(kw_sorted) if chunk_id == answer_id),
            None,
        )

        # 2.向量路：问题向量化后全量 query，找答案块名次
        query_vec = embed([q["问题"]])[0]
        vec_res = collection.query(query_embeddings=[query_vec], n_results=TOTAL)
        vec_pos = next(
            (i + 1 for i, chunk_id in enumerate(vec_res["ids"][0]) if chunk_id == answer_id),
            None,
        )

        # 3.混合路：候选池与返回都放全量，按文件名+小节匹配答案块名次
        hybrid_evidence = hybrid_search_knowledge_base(
            q["问题"], candidate_size=TOTAL, n_results=TOTAL
        )
        hybrid_pos = next(
            (
                e["混合名次"]
                for e in hybrid_evidence
                if e["文件名"] == answer_block["文件名"]
                and f"## {answer_block['小节']}" in e["文件内容"]
            ),
            None,
        )

        rows.append(
            {
                "编号": q["编号"],
                "类别": q["类别"],
                "问题": q["问题"],
                "关键词名次": kw_pos,
                "向量名次": vec_pos,
                "混合名次": hybrid_pos,
            }
        )
        print(
            f"{q['编号']:<4}{q['类别']:<6}{str(kw_pos):<7}{str(vec_pos):<7}{str(hybrid_pos):<7}{q['问题'][:20]}"
        )

    RANKS_FILE.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n三路名次已存 {RANKS_FILE.name}")


if __name__ == "__main__":
    main()
