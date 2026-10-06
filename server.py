from fastapi import FastAPI
from agent import run_agent
from pydantic import BaseModel

app = FastAPI()


class Message(BaseModel):
    mes: str


@app.post("/ask")
def ask(question: Message):
    q = question.mes
    if not q.strip():
        return {"错误": "问题不能为空"}
    answer, records = run_agent(q)
    SOURCE_MAP = {
        "search_knowledge": "检索",
        "web_search": "搜索",
        "calculate": "计算",
    }
    for rec in records:
        source = SOURCE_MAP.get(rec["name"])
        if source is None or rec["evidence"] is None:
            continue
        for r in rec["evidence"]:
            r["来源"] = source
    return {"回答": answer, "回答调用记录": records}
