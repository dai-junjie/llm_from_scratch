from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
load_dotenv()

mcp = FastMCP(
    name="caclulator",
    host="0.0.0.0",
    port=8050
)

@mcp.tool(name="func_add")
def add(a:int,b:int)->int:
    """add two numbers"""
    return a+b



if __name__ == "__main__":
    transport = "sse"
    # transport = "stdio"
    if transport == "stdio":
        print("running server with stdio transport")
        mcp.run(transport="stdio")
    elif transport == "sse":
        print("running server with sse transport")
        mcp.run(transport="sse")
        
