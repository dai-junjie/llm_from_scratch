from typing import TypedDict
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import Annotated
from langgraph.graph.message import add_messages
import os

class PersonDictionary(TypedDict):
    name:str
    age:int
    is_student:bool
    
our_person:PersonDictionary = {
    "name":"dai",
    "age":18,
    "is_student":True
}
print(our_person)

class AgentState(TypedDict):
    user_message:Annotated[list, add_messages]
    
llm = ChatOpenAI(
    model_name="qwen-plus-latest",
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY")
)


def first_node(state:AgentState)->AgentState:
    resp = llm.invoke(state["user_message"])
    print(f'\nAI:{resp.content}')
    return {
        "user_message":[resp]
    }
# building graph
graph = StateGraph(AgentState)
graph.add_node('node',first_node)
graph.add_edge(START, 'node')
graph.add_edge('node', END)

# memory 
memory = InMemorySaver()
config = {
    "thread_id": "agent-1",
}
agent = graph.compile(checkpointer=memory)

# state:AgentState = {
#     "user_message": HumanMessage(content="Hello, who are you?") # 这里的内容只能是str之类的，不能是HumanMessage类型
# }

# # running agent
# resp = agent.invoke(
#     state
# )

# print(f'\nFinal response: {resp}')

while True:
    user_input =input("Enter: ")
    if user_input.lower() == 'exit':
        break
    agent.invoke({
        "user_message":[HumanMessage(content=user_input)],
        
    },config=config)