from langchain_openai import ChatOpenAI
import os
llm = ChatOpenAI(
    model="qwen3-coder-480b-a35b-instruct",
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
    streaming=True
)


for chunk in llm.stream("写一个函数，能够把python的函数封装为schema，使得函数的功能，参数和返回值都能够被描述"):
    print(chunk.content, end="")
