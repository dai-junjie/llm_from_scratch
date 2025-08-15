"""
 aim:
  1. 理解tools和条件边是什么
  2. 理解BaseMessage
  3. 理解Annotated Type Annotation
  4. 如何访问in-built tools in langgraph
"""
from typing import TypedDict,Annotated,Sequence
from langchain_core.messages import BaseMessage,AIMessage,HumanMessage,ToolMessage
from langchain_openai import ChatOpenAI, OpenAI
from langgraph.graph import StateGraph,END,START
from langgraph.graph.message import add_messages
from langchain_core.tools import Tool
from langgraph.prebuilt import ToolNode
from langchain_tavily import TavilySearch
import os

x:Annotated[int,"age"] = 20

print(x)

numbers:Sequence[int] = [1,2,3]

print(numbers)

class DickMessage(BaseMessage):
    role:str = "dick"
    
dickMessage = DickMessage(content="hello",type="dickType")
dickMessage.additional_kwargs = {
    "dick_num":123
}
# dickMessage.pretty_print()

aiMessage = AIMessage(content="我是一个AI")
print(aiMessage.type)
# aiMessage.pretty_print()

llm = ChatOpenAI(
    model_name="qwen-plus-latest",
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY")
)

tavilySearch = TavilySearch()

def news_tool_func(query:str):
    # 这个是工具的描述，会被大模型用来理解工具的功能，如果不写的话，需要用Tool包装
    """查询重大新闻摘要""" 
    return "南工程团队成功研制出让母猪长出翅膀飞上蓝天的技术"

news_tool = Tool(
    name="news_tool",
    description="获取重大新闻摘要，可与搜索工具同时使用",
    func=news_tool_func
)

tools = [tavilySearch,news_tool]
llm = llm.bind_tools(tools) # 绑定了，告诉大模型这个工具的知识
# resp = tavilySearch.invoke(
#     query
# )

# print(f'\n tavily resp:{resp}')

class State(TypedDict):
    messages:Annotated[list,add_messages]


def should_continue(state:State):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool"
    else:
        return "end"
    
def chatbot(state:State):
    resp = llm.invoke(state["messages"])
    return {
        "messages":[resp]
    }
    
graph = StateGraph(State)
graph.add_node("chatbot",chatbot)
graph.add_node("tools",ToolNode(tools))
graph.add_edge(START,"chatbot")

graph.add_edge("tools","chatbot")

graph.add_conditional_edges(
    "chatbot",
    should_continue,
    {
        "tool":"tools",
        "end":END
    }
)

agent = graph.compile()


query = "请同时使用tavily_search搜索今天的新闻，并使用news_tool获取重大新闻摘要"

for event in agent.stream({
    "messages":[HumanMessage(content=query)],
    },stream_mode="values"):
    if "messages" in event:
        event["messages"][-1].pretty_print()
        # if isinstance(event["messages"][-1],AIMessage):
        #     print(event["messages"][-1].tool_calls)
        #     break
        # print(event["messages"][-1])