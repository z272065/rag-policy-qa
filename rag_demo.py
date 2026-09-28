"""RAG demo 第一步：把 docs 目录下的所有文档读进来。

不管你在哪个目录下运行，都能正确找到 docs：
    cd 练习/rag && python rag_demo.py      -> 可以
    cd 练习     && python rag/rag_demo.py  -> 也可以
"""

import os
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.environ["DEEPSEEK_API_KEY"],
)


# ---- 1) 定位 docs 目录 -------------------------------------------------
# Path(__file__)   当前脚本自己的路径（__file__ 是 Python 注入的变量）
# .resolve()       转成绝对路径，顺便解析 .. 和软链接
# .parent          去掉文件名，拿到脚本所在目录（也就是 rag/）
# / "docs"         用 / 运算符拼上子目录
BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"

if not DOCS_DIR.is_dir():
    raise SystemExit(f"找不到目录: {DOCS_DIR}（当前工作目录是 {Path.cwd()}）")

# ---- 2) 找出 docs 下的所有 .md 文件 -------------------------------------
# glob("*.md")  只匹配直接子项，不递归（要递归用 rglob）
# 它返回的是生成器，只能遍历一次，所以这里用 sorted() 立刻转成列表，
# 顺便保证 01 -> 04 的顺序（文件系统返回顺序是不保证的）
doc_files = sorted(DOCS_DIR.glob("*.md"))

print(f"docs 目录 : {DOCS_DIR}")
print(f"找到文件  : {len(doc_files)} 个\n")


def score(question, text):
    score = 0
    for k in range(len(question) - 1):
        if question[k : k + 2] in text:
            score += 1
    return score


def run_agent(user_prompt):
    # ---- 3) 逐个读取 -------------------------------------------------------
    chunks = []
    for i, path in enumerate(doc_files, start=1):
        # read_text 必须显式指定 encoding，Windows 默认编码读中文会报 UnicodeDecodeError
        text = path.read_text(encoding="utf-8")

        # path.name        文件名（带扩展名）
        # path.stem        文件名（不带扩展名），后面做 doc_id 可以用

        # for chunk in text.split("\n## "):
        #     chu = "## " + chunk.strip()
        #     chunks.append({"文件序号": i, "文件名": path.name, "文件内容": chu})
        #     # print(chu)
        # # print(chus)
        # print(chunks)

        parts = text.split("\n## ")  # 先存成变量，下面要回头拿 parts[0]

        for j, chunk in enumerate(parts):
            if j == 0:
                continue

            body = "## " + chunk.strip()  # 补回被刀吃掉的 "## "

            if j == 1:  # 第一小节：把一级标题拼到它最前面
                body = parts[0].strip() + "\n\n" + body

            chunks.append({"文件序号": i, "文件名": path.name, "文件内容": body})
    for cc in chunks:
        cc_score = score(user_prompt, cc["文件内容"])
        cc["分数"] = cc_score
        print(
            f"分数:{cc_score},文件名:{cc['文件名']},内容开头:{cc['文件内容'][:30].replace('\n', ' ')}"
        )

    chunks.sort(key=lambda cc: cc["分数"], reverse=True)
    # print(chunks)
    top3 = chunks[:3]
    # top_content = []
    # for top in top3:
    #     top_content += {top["文件名"], top["文件内容"]}
    top_content = ""  # ← 空字符串打底
    for top in top3:
        top_content += (
            f"【{top['文件名']}】\n{top['文件内容']}\n\n"  # ← 每轮拼上去的是一段文字
        )
    prompt = f"{top_content}\n{user_prompt}"
    messages = [
        {
            "role": "system",
            "content": "你是一个hr助手,需要根据材料如实回答,不清楚不确定的问题绝对不能编",
        },
        {"role": "user", "content": prompt},
    ]
    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=messages,
    )
    message = response.choices[0].message
    messages.append(message)
    return message.content

    # for chu in chunks:
    #     print(chu["文件名"], chu["文件内容"][:30].replace("\n", " "))
    # print(f"总共{len(chunks)}块")


if __name__ == "__main__":
    print(run_agent("我要休假, 需要提前几天申请"))

# print(run_agent("出差住宿超过 500 元的部分，公司给报销吗？"))
# print(run_agent("个人娱乐消费能报销吗"))
# for i in range(5):
#     print(f"===== 第 {i + 1} 次 =====")
#     print(run_agent("个人娱乐消费能报销吗"))
# print(run_agent("个人娱乐消费能报销吗"))
