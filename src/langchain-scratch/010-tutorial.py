import os
from langchain_openai import ChatOpenAI  # 或者根据您使用的模型选择正确的导入
from langchain_core.messages import HumanMessage, SystemMessage
# LLM模型
model = ChatOpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key = os.getenv("QWQ_API_KEY"),
        model="qwq-plus-latest",
        streaming=True,
        temperature=0.5,
        top_p = 0.5,
        )

from langchain.schema.messages import HumanMessage, SystemMessage
messages = [
    SystemMessage(content="You are Micheal Jordan."),
    HumanMessage(content="Which shoe manufacturer are you associated with?"),
]
for chunk in  model.stream(messages):
    print(chunk.content,end='',flush=True)