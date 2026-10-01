import os
from openai import OpenAI
from dotenv import load_dotenv
from retrieval import search_knowledge_base

load_dotenv()

client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.environ["DEEPSEEK_API_KEY"],
)


def run_agent(user_prompt):
    top_content = search_knowledge_base(user_prompt)
    # 4. 拼 prompt 调大模型 —— 从这里开始和关键词版逐行相同
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


if __name__ == "__main__":
    print(run_agent("我想休假，要提前几天申请？"))
