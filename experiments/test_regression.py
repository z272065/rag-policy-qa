# -*- coding: utf-8 -*-
"""路由回归测试：用 eval20_v1.json 考卷对 /ask 的实际响应做自动断言。

环境与数据对象
- 依赖本地服务先启动：python server.py（默认 http://127.0.0.1:8000）
- 考卷：同目录 eval20_v1.json（20 题：库内 10 + 需搜索 5 + 库外 5）
- 检索数据：当前 chroma_db（由 build_db.py 生成）
- 判定口径沿用考卷文件头：按最终走向判，中途绕路不扣分

跑法（在 rag 目录）
    python -m pytest experiments/test_regression.py -v

注意：模型输出有随机性，单题偶发失败先重跑确认是稳定失败还是随机波动，
不要一次失败就断定代码有问题。
"""

import json
from pathlib import Path

import pytest
import requests

BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 300  # 单题 agent 可能多轮调用，与 eval_run_v1.py 保持同一上限

EXPERIMENTS_DIR = Path(__file__).resolve().parent
EVAL_FILE = EXPERIMENTS_DIR / "eval20_v1.json"

# 拒答措辞标志，参考 期望_拒答.json 的样例
REFUSE_MARKERS = ("没有", "查无", "咨询 HR", "咨询HR")


def load_questions():
    data = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    return data["题目"]


QUESTIONS = load_questions()


def ask(question):
    """打 /ask，返回解析后的 JSON（回答 + 路由记录）。"""
    resp = requests.post(
        f"{BASE_URL}/ask", json={"mes": question}, timeout=TIMEOUT
    )
    resp.raise_for_status()
    return resp.json()


@pytest.fixture(scope="module")
def service_available():
    """确认本地服务在线；空问题按 server.py 逻辑应返 422。"""
    try:
        probe = requests.post(
            f"{BASE_URL}/ask", json={"mes": " "}, timeout=10
        )
        assert probe.status_code == 422
    except Exception as exc:
        pytest.exit(
            f"本地 /ask 服务不可用，请先在 rag 目录运行 python server.py：{exc}"
        )


@pytest.mark.parametrize(
    "q",
    QUESTIONS,
    ids=[f"题{q['编号']}-{q['类别']}" for q in QUESTIONS],
)
def test_route(service_available, q):
    data = ask(q["问题"])
    tool_names = [record["name"] for record in data["路由"]]
    answer = data["回答"]
    expected = q["正确路由"]

    if expected == "检索":
        assert "search_knowledge" in tool_names, (
            f"题{q['编号']}预期检索制度库，实际工具链：{tool_names}"
        )

    elif expected == "检索+计算":  # 题10：病假工资，检索条款 + 本地计算
        assert "search_knowledge" in tool_names, (
            f"题{q['编号']}未检索制度库，实际工具链：{tool_names}"
        )
        assert "calculate" in tool_names, (
            f"题{q['编号']}未调用计算器，实际工具链：{tool_names}"
        )
        # 概率性断言：模型措辞可能变化，失败先重跑确认
        assert "9600" in answer, f"题{q['编号']}回答未含预期数字 9600：{answer}"

    elif expected == "搜索":
        assert "web_search" in tool_names, (
            f"题{q['编号']}预期上网搜索，实际工具链：{tool_names}"
        )

    elif expected == "拒答":
        assert any(marker in answer for marker in REFUSE_MARKERS), (
            f"题{q['编号']}预期拒答（含拒答措辞），实际回答：{answer}"
        )

    else:
        pytest.fail(f"考卷出现未知的正确路由类型：{expected}")
