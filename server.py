from agent import run_agent
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


class Message(BaseModel):
    mes: str


def extract_section(content):
    """从检索块内容里提取第一个二级标题（## 小节名）作为条款名。

    块按二级标题切分，没有 ## 标题时返回空串。
    """
    for line in content.splitlines():
        if line.startswith("## "):
            return line[3:].strip()
    return ""


@app.post("/ask")
def ask(question: Message):
    q = question.mes
    if not q.strip():
        raise HTTPException(status_code=422, detail="问题不能为空")
    answer, records = run_agent(q)
    SOURCE_MAP = {
        "search_knowledge": "检索",
        "web_search": "搜索",
        "calculate": "计算",
    }
    # citations：引用编号 -> 文件名 + 小节名，只收集制度库检索块
    citations = {}
    for rec in records:
        source = SOURCE_MAP.get(rec["name"])
        if source is None or rec["evidence"] is None:
            continue
        for r in rec["evidence"]:
            r["来源"] = source
            if rec["name"] == "search_knowledge" and r.get("引用编号") is not None:
                citations[str(r["引用编号"])] = {
                    "文件名": r["文件名"],
                    "小节": extract_section(r["文件内容"]),
                }
    return {"回答": answer, "路由": records, "引用来源": citations}


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
