import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from agent import run_agent


EVAL_FILE = Path(__file__).resolve().parent / "eval20_v1.json"
RESULTS_FILE = Path(__file__).resolve().parent / "eval20_results_raw_v1.json"


questions = json.loads(EVAL_FILE.read_text(encoding="utf-8"))["题目"]
result = []
for q in questions:
    try:
        answer, calls = run_agent(q["问题"], max_turns=5)
        result.append(
            {
                "编号": q["编号"],
                "类别": q["类别"],
                "问题": q["问题"],
                "标准答案": q["标准答案"],
                "回答": answer,
                "正确路由": q["正确路由"],
                "路由": calls,
            }
        )
        print(f"第{q['编号']}题完成")
    except Exception as e:
        result.append(
            {
                "编号": q["编号"],
                "类别": q["类别"],
                "问题": q["问题"],
                "标准答案": q["标准答案"],
                "回答": repr(e),
                "正确路由": q["正确路由"],
                "路由": [],
            }
        )
        print(f"第{q['编号']}题发生错误")
RESULTS_FILE.write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
)
