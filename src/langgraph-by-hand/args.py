import datetime
from typing import List


def func(username:str,email:str,age:int):
    print(username,email,age)
    
input_val = {
    "username":'daijunjie',
    "email":'daijunjie@example.com'
}

func(**input_val,age=24)


def extract_info(uname:str,email:str,age:int):
    print(f'用户名:{uname},邮箱:{email},年龄:{age}')

input_val = {
    "email":"daijunjie@example.com",
    "age":24
}

extract_info("djj",**input_val)
print('='*100)
# 模拟工具调用

def web_tool(query:str|List[str]):
    if isinstance(query,list):
        results = []
        for q in query:
            results.append(f'query:{q}')
        return results
    else:
        return query

def weather_tool(date:int):
    return f'天气信息:{date}'

query = ["你好","你好"]
date = datetime.date.today()

tool_dict = {
    "web_tool":web_tool,
    "weather_tool":weather_tool
}

tool_calls = [
    {"tool_name":"web_tool","args":{"query":query}},
    {"tool_name":"weather_tool","args":{"date":date}}
]

for tool_call in tool_calls:
    tool_name = tool_call["tool_name"]
    tool_args = tool_call["args"]
    tool_func = tool_dict[tool_name]
    print(tool_func(**tool_args))
