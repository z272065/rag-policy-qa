from agent import run_agent
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

app = FastAPI()


class Message(BaseModel):
    mes: str


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
    for rec in records:
        source = SOURCE_MAP.get(rec["name"])
        if source is None or rec["evidence"] is None:
            continue
        for r in rec["evidence"]:
            r["来源"] = source
    return {"回答": answer, "路由": records}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
