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
            "description": "检索公司内部制度库。仅当问题问的是公司自己制定的制度、流程、标准、额度（考勤、请假、报销、福利等）时调用；国家法律法规和公开政策不在库里，不要用本工具。",
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
            "description": "搜索互联网上的公开信息。当问题涉及国家法律法规、公共政策（社保公积金、产假、年假、试用期工资等国家有统一规定的内容）或时效性信息（当年节假日安排等）时调用；公司制度库里查不到、但属于公开领域的问题，也用它。公司自己的内部制度上网搜不到，要用 search_knowledge。",
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
            "description": "精确计算算术表达式，返回数值结果。只要回答中需要给出计算得出的数字（金额、天数、比例、折扣等），一律调用本工具，不要自己心算；即使用户没有明确要求计算也要调用。",
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
    evidence = search_knowledge_base(query)
    top_content = ""
    for evi in evidence:
        top_content += f"【{evi['文件名']}】\n{evi['文件内容']}\n\n"
    return (top_content, evidence)


def web_search(query):
    print(f"web_search被调用了,query:{query}")
    resp = tavily_client.search(query, max_results=5)
    lines = [f"【{r['title']}】({r['url']})\n{r['content']}" for r in resp["results"]]
    evidence = []
    for r in resp["results"]:
        evidence.append({"标题": r["title"], "URL": r["url"], "内容": r["content"]})
    return ("\n\n".join(lines), evidence)


def calculate(expression):
    print(f"calculate被调用了,expression:{expression}")
    evidence = []
    for e in expression:
        if e not in "0123456789+-*/().% ":
            return ("只支持纯算术表达式", evidence)
    result = eval(expression)
    evidence.append({"表达式": expression, "结果": result})
    return (str(result), evidence)


TOOL_BAG = {
    "search_knowledge": search_knowledge,
    "web_search": web_search,
    "calculate": calculate,
}


def run_agent(user_prompt, max_turns=5):
    messages = [
        {
            "role": "system",
            "content": "你有公司制度库可查，也可以搜索公开网页。回答公司制度问题必须基于查到的材料。制度库里没有的，先判断：国家法规、公开政策、时效信息→用 web_search 查了再答；公司自主事项（年终奖、班车这类网上也查不到的）→如实告知查无并建议咨询 HR。无论哪条路，都不编造。",
        },
        {"role": "user", "content": user_prompt},
    ]
    tool_call_list = []
    texts = []

    for _ in range(max_turns):
        response = client.chat.completions.create(
            model="deepseek-flash", messages=messages, tools=tools
        )
        message = response.choices[0].message
        messages.append(message)
        if message.content:
            texts.append(message.content)
        if not message.tool_calls:
            return "\n\n".join(texts), tool_call_list

        for tool_call in message.tool_calls:
            evidence = None
            func_args = None
            try:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments or "{}")
                if func_name not in TOOL_BAG:
                    result = f"{func_name}不存在,重试"
                elif not isinstance(func_args, dict):
                    result = "参数必须是JSON对象,重试"
                else:
                    result, evidence = TOOL_BAG[func_name](**func_args)
            except json.JSONDecodeError as e:
                result = f"参数不是合法JSON:{e}。你刚才生成的是:{tool_call.function.arguments!r},请重新生成"
            except Exception as e:
                result = f"工具执行出错:{e},请修正后重试"

            messages.append(
                {"role": "tool", "content": result, "tool_call_id": tool_call.id}
            )
            tool_call_list.append(
                {"name": func_name, "arguments": func_args, "evidence": evidence}
            )
    return "超出最大轮次,未获取内容", tool_call_list


if __name__ == "__main__":
    print(run_agent("1024的37%是多少"))
    print(run_agent("我月薪 12000，请一个月病假，工资发多少"))
