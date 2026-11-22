# -*- coding: utf-8 -*-
"""混合检索时代完整评测：20 题打 /ask，落机器稿并初始化人工判定表。

环境与数据对象
- 依赖本地服务先启动：python server.py（http://127.0.0.1:8000）
- 考卷：eval20_v1.json（库内 10 + 需搜索 5 + 库外 5）
- 检索：当前 chroma_db（209 块）+ 关键词向量混合检索

跑法（在 rag 目录）
    python experiments/eval_run_hybrid.py

产物
- eval20_results_raw_hybrid.json：机器稿，每次运行覆盖
- eval20_results_hybrid.json：人工判定表，仅在不存在时创建，已存在不覆盖
"""

import json
from pathlib import Path

import requests

BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 300  # 单题 agent 可能多轮调用，与 eval_run_v1.py 保持同一上限

EXPERIMENTS_DIR = Path(__file__).resolve().parent
EVAL_FILE = EXPERIMENTS_DIR / "eval20_v1.json"
RAW_FILE = EXPERIMENTS_DIR / "eval20_results_raw_hybrid.json"
JUDGE_FILE = EXPERIMENTS_DIR / "eval20_results_hybrid.json"


def ask(question):
    """打 /ask，返回解析后的 JSON（回答 + 路由）。"""
    resp = requests.post(f"{BASE_URL}/ask", json={"mes": question}, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def main():
    questions = json.loads(EVAL_FILE.read_text(encoding="utf-8"))["题目"]

    raw_rows = []
    judge_rows = []
    for q in questions:
        data = ask(q["问题"])
        base = {
            "编号": q["编号"],
            "类别": q["类别"],
            "问题": q["问题"],
            "标准答案": q["标准答案"],
            "回答": data["回答"],
            "正确路由": q["正确路由"],
            "路由": data["路由"],
        }
        raw_rows.append(base)

        # 人工判定表：拷贝完整数据，判定字段留空待人工填
        judge_row = dict(base)
        judge_row.update(
            {"路由判定": "", "路由错误方向": "", "回答判定": "", "备注": ""}
        )
        judge_rows.append(judge_row)
        print(f"第{q['编号']}题完成")

    RAW_FILE.write_text(
        json.dumps(raw_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n机器稿已存 {RAW_FILE.name}")

    if JUDGE_FILE.exists():
        print(f"人工判定表 {JUDGE_FILE.name} 已存在，未覆盖，请在原文件上继续判定")
    else:
        JUDGE_FILE.write_text(
            json.dumps(judge_rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"人工判定表已初始化 {JUDGE_FILE.name}，判定字段留空待填")


if __name__ == "__main__":
    main()
