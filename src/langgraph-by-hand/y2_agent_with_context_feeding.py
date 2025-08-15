import os
from typing import TypedDict,List,Union
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph,START,END

def myfunc(value:Union[int,float]):
    return value

def myfunc2(value: int | str | bool):
    return value

# print(myfunc2(False))

class State(TypedDict):
    messages:List[Union[HumanMessage,AIMessage,SystemMessage]]

llm = ChatOpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
    model_name="qwen-plus-latest",
)

conversation_history = [
    SystemMessage(content="你是一个说话风格像海盗的AI助手，你会根据用户的问题，提供专业的答案。"),
]

def our_processing_node(state:State):
    resp = llm.invoke(state["messages"])
    state['messages'] += [AIMessage(content=resp.content)]
    return state

graph = StateGraph(State)
graph.add_node('node',our_processing_node)
graph.add_edge(START,'node')
graph.add_edge('node',END)

agent = graph.compile()


while True:
    user_input = input("Human: ")
    if user_input.lower() == "exit":
        break
    conversation_history += [HumanMessage(content=user_input)]
    for event in agent.stream(
        {
            "messages": conversation_history,
        }
    ):
        for value in event.values():
            print(f'len messages: {len(value["messages"])}')
            value['messages'][-1].pretty_print()
            conversation_history = value['messages']