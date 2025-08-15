from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage
import os

class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_name: str
    user_email: str

# 测试：返回完整的旧state
def chatbot_return_full_state(state: State):
    print(f"输入state: {state}")
    
    # 模拟AI回答
    ai_response = AIMessage(content="这是AI的回答")
    
    # 返回完整的旧state + 新的AI消息
    new_state = state.copy()
    new_state["messages"] = state["messages"] + [ai_response]
    
    print(f"返回state: {new_state}")
    return new_state

# 构建图
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot_return_full_state)
graph_builder.set_entry_point("chatbot")
graph_builder.add_edge("chatbot", END)

# 添加内存
memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "test"}}

print("=== 第一次调用 ===")
result1 = graph.invoke({
    "messages": [HumanMessage(content="第一条消息")],
    "user_name": "张三",
    "user_email": "zhang@example.com"
}, config=config)

print(f"第一次结果: {result1}")

print("\n=== 第二次调用 ===")
result2 = graph.invoke({
    "messages": [HumanMessage(content="第二条消息")],
}, config=config)

print(f"第二次结果: {result2}")
print(f"Messages数量: {len(result2['messages'])}")