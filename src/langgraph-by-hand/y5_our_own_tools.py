from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
# from langchain_tavily import TavilySearch
from datetime import datetime
import os

@tool
def weather_tool(city:str)->str:
    """Get the weather in a given city"""
    return f"the weather in {city} is 天上下核弹"

@tool
def social_media_follower_count(social_media:str)->str:
    """
    获取社交媒体平台的关注者数量
    Args:
        social_media: 社交媒体平台名称，如twitter、facebook、instagram等
    Returns:
        社交媒体平台的关注者数量
    Example:
        >>> social_media_follower_count("twitter")
        '1234 followers'
    """
    return f"you have 9876 followers on {social_media}"

tools = [weather_tool,social_media_follower_count]

llm = ChatOpenAI(
    model_name="qwen-max-latest",
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY")
)

agent = create_react_agent(
    model=llm,
    tools=tools,
    name="weather_agent",
    prompt="""You are my ai assistant that has access to certain tools.
    use the tools to help me with my tasks.工具请同时使用
    """    ,
)

for event in agent.stream({
    "messages":[HumanMessage(content="请查询西京的天气,查询我在twitter和facebook平台的follower数量")]
    },
                          stream_mode="values"
    ):
    event['messages'][-1].pretty_print()