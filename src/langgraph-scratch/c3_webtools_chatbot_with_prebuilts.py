from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langchain_tavily import TavilySearch
from typing import Annotated
from langchain_openai import ChatOpenAI
import os
llm = ChatOpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
    model='qwen-plus-latest',
    streaming=True
)


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

tool = TavilySearch(max_results=2)
tools = [tool]
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()


# 调用,其实每次人机对话的记忆都是新的，没有维护之前的人机对话

def stream_graph_updates(user_input: str):
    # 用户一次输入，这里会调用图模型，可能会多次执行，这里stream函数的返回是一个迭代对象。
    # 比如需要使用工具，第一个event是工具调用，第二个是使用工具调用结果来生成回答。
    # 所以这里assistant第一次输出是工具调用的结果，第二次输出是生成的回答。
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
