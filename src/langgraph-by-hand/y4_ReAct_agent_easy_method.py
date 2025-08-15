from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
# from langgraph.graph import START,END,StateGraph
from langchain_tavily import TavilySearch
import os

tools = [
    TavilySearch(max_results=2)
]

llm = ChatOpenAI(
    model_name="qwen-turbo-latest",
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY")
)

agent = create_react_agent(
    model=llm,
    tools = tools,
    name='search_agent',
    prompt="""You are my ai assistant that has access to certain tools.
    use the tools to help me with my tasks
    """
)

result = agent.stream({
    "messages":[HumanMessage(content="请搜索今天的新闻")]
},stream_mode="values")

for event in result:
    event['messages'][-1].pretty_print()