import os
import json
from openai import OpenAI
from tavily import TavilyClient
from dotenv import load_dotenv
from retrieval import search_knowledge_base


load_dotenv()

client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.environ["DEEPSEEK_API_KEY"],
)

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "需要根据材料回答问题的时候调用",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要检索的用户问题"}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "需要时效性信息或公司制度库以外的公开信息时调用,公司内部规定不在公开网页上",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要搜索的问题或关键词"}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "对纯数字的算术表达式做四则运算；需要根据已知数字算出具体结果（钱、天数、比例）时调用",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "纯算术表达式，如 12000*0.8",
                    },
                },
            },
        },
    },
]


def search_knowledge(query):
    print(f"search_knowledge被调用了,query:{query}")
    return search_knowledge_base(query)


def web_search(query):
    print(f"web_search被调用了,query:{query}")
    resp = tavily_client.search(query, max_results=5)
    lines = [f"【{r['title']}】({r['url']})\n{r['content']}" for r in resp["results"]]
    return "\n\n".join(lines)


def calculate(expression):
    print(f"calculate被调用了,expression:{expression}")
    for e in expression:
        if e not in "0123456789+-*/().% ":
            return "只支持纯算术表达式"
    return str(eval(expression))


TOOL_BAG = {
    "search_knowledge": search_knowledge,
    "web_search": web_search,
    "calculate": calculate,
}


def run_agent(user_prompt, max_turns=5):
    messages = [
        {
            "role": "system",
            "content": "你有公司制度库可以查；回答必须基于查到的材料；材料里没有就直说不知道，绝不编",
        },
        {"role": "user", "content": user_prompt},
    ]
    for _ in range(max_turns):
        response = client.chat.completions.create(
            model="deepseek-flash", messages=messages, tools=tools
        )
        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            return message.content
        for tool_call in message.tool_calls:
            try:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments or "{}")
                if func_name not in TOOL_BAG:
                    result = f"{func_name}不存在,重试"
                elif not isinstance(func_args, dict):
                    result = "参数必须是JSON对象,重试"
                else:
                    result = TOOL_BAG[func_name](**func_args)
            except json.JSONDecodeError as e:
                result = f"参数不是合法JSON:{e}。你刚才生成的是:{tool_call.function.arguments!r},请重新生成"
            except Exception as e:
                result = f"工具执行出错:{e},请修正后重试"

            messages.append(
                {"role": "tool", "content": result, "tool_call_id": tool_call.id}
            )
    return "超出最大轮次,未获取内容"


if __name__ == "__main__":
    print(run_agent("1024的37%是多少"))
    print(run_agent("我月薪 12000，请一个月病假，工资发多少"))
