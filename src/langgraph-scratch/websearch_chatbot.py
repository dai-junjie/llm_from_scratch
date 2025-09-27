from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.tools import tool

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt

import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import asyncio
import json
from typing import Optional

# Initialize the LLM with specified configuration
llm = ChatOpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
    model='qwen-plus-latest',
    streaming=True
)

# Define the state structure
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Create the graph
graph_builder = StateGraph(State)

# Define human assistance tool
@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]

# Initialize tools
search_tool = TavilySearch(max_results=2)
tools = [search_tool, human_assistance]
llm_with_tools = llm.bind_tools(tools)

# Define the chatbot node
def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    # Disable parallel tool calling to avoid repeating tool invocations when we resume
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}

# Add nodes to the graph
graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# Add edges to the graph
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# Add memory
memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# Function to stream graph updates
def stream_graph_updates(user_input: str, thread_id: str = "1"):
    config = {"configurable": {"thread_id": thread_id}}
    
    # Stream events from the graph
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )
    
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()
    
    # Check if the graph is interrupted and needs human input
    snapshot = graph.get_state(config)
    if snapshot.next:
        # Graph is interrupted, waiting for human input
        return snapshot
    
    return None

# Function to provide human response when the graph is interrupted
def provide_human_response(human_response: str, thread_id: str = "1"):
    config = {"configurable": {"thread_id": thread_id}}
    
    # Create a command to resume the graph with human response
    human_command = Command(resume={"data": human_response})
    
    # Stream events from the graph with human response
    events = graph.stream(human_command, config, stream_mode="values")
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()

# Pydantic models for API
class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = "default"

class HumanResponse(BaseModel):
    response: str
    thread_id: Optional[str] = "default"

# FastAPI app
app = FastAPI(title="Web Search Chatbot API", version="1.0.0")

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Chat with the AI assistant that can search the web and request human help."""
    try:
        config = {"configurable": {"thread_id": request.thread_id}}
        
        # Check if there's an existing interrupted state
        snapshot = graph.get_state(config)
        if snapshot.next:
            return {
                "status": "interrupted",
                "message": "There is a pending human assistance request. Please use /human_response endpoint first.",
                "thread_id": request.thread_id
            }
        
        messages = []
        interruption_data = None
        
        # Stream events from the graph
        events = graph.stream(
            {"messages": [{"role": "user", "content": request.message}]},
            config,
            stream_mode="values",
        )
        
        for event in events:
            if "messages" in event:
                messages.append(event["messages"][-1])
        
        # Check if the graph is interrupted
        snapshot = graph.get_state(config)
        if snapshot.next:
            # Extract interruption data if available
            if snapshot.tasks and len(snapshot.tasks) > 0:
                task = snapshot.tasks[0]
                if hasattr(task, 'interrupts') and task.interrupts:
                    interruption_data = task.interrupts[0]
            
            return {
                "status": "interrupted",
                "messages": [{"role": msg.type, "content": str(msg.content)} for msg in messages],
                "interruption_query": interruption_data.get("query") if interruption_data else "Human assistance needed",
                "thread_id": request.thread_id
            }
        
        return {
            "status": "completed",
            "messages": [{"role": msg.type, "content": str(msg.content)} for msg in messages],
            "thread_id": request.thread_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/human_response")
async def human_response_endpoint(request: HumanResponse):
    """Provide human response to continue interrupted conversation."""
    try:
        config = {"configurable": {"thread_id": request.thread_id}}
        
        # Check if there's actually an interrupted state
        snapshot = graph.get_state(config)
        if not snapshot.next:
            return {
                "status": "error",
                "message": "No pending human assistance request found.",
                "thread_id": request.thread_id
            }
        
        messages = []
        
        # Create a command to resume the graph with human response
        human_command = Command(resume={"data": request.response})
        
        # Stream events from the graph with human response
        events = graph.stream(human_command, config, stream_mode="values")
        for event in events:
            if "messages" in event:
                messages.extend(event["messages"])
        
        # Check if completed or interrupted again
        snapshot = graph.get_state(config)
        if snapshot.next:
            return {
                "status": "interrupted",
                "messages": [{"role": msg.type, "content": str(msg.content)} for msg in messages],
                "interruption_query": "Human assistance needed again",
                "thread_id": request.thread_id
            }
        
        return {
            "status": "completed",
            "messages": [{"role": msg.type, "content": str(msg.content)} for msg in messages],
            "thread_id": request.thread_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/threads/{thread_id}/state")
async def get_thread_state(thread_id: str):
    """Get the current state of a conversation thread."""
    try:
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = graph.get_state(config)
        
        return {
            "thread_id": thread_id,
            "is_interrupted": bool(snapshot.next),
            "message_count": len(snapshot.values.get("messages", [])),
            "last_messages": [
                {"role": msg.type, "content": str(msg.content)} 
                for msg in reversed(snapshot.values.get("messages", []))
            ] if snapshot.values.get("messages") else []
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/threads/{thread_id}")
async def clear_thread(thread_id: str):
    """Clear a conversation thread."""
    try:
        config = {"configurable": {"thread_id": thread_id}}
        # Clear the thread by creating a new empty state
        graph.update_state(config, {"messages": []})
        
        return {
            "status": "success",
            "message": f"Thread {thread_id} cleared successfully",
            "thread_id": thread_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server."""
    uvicorn.run(app, host=host, port=port)

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        # Run web server
        print("Starting web server on http://0.0.0.0:8000")
        print("API docs available at http://0.0.0.0:8000/docs")
        run_server()
    else:
        # Run CLI interface
        print("Start chatting with the AI! Type 'quit' to exit.")
        print("Add 'server' argument to run web server mode.")
        thread_id = "1"
        
        while True:
            try:
                user_input = input("User: ")
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break
                
                # Process user input
                snapshot = stream_graph_updates(user_input, thread_id)
                
                # If the graph is interrupted, ask for human assistance
                if snapshot and snapshot.next:
                    print("\nThe AI needs human assistance. Please provide your response.")
                    human_input = input("Human: ")
                    provide_human_response(human_input, thread_id)
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                break