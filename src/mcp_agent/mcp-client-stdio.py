import asyncio
import nest_asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

nest_asyncio.apply()  # Needed to run interactive python


async def main():
    # Define server parameters
    server_params = StdioServerParameters(
        command="python",  # The command to run your server
        args=["src/mcp_agent/mcp-simple-server.py"],  # Arguments to the command
        transport="stdio",
    )

    # Connect to the server
    # 创建一个client会话来与server进行交互,server_params可以用来启动server
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the connection
            await session.initialize()
            """
            session可以用来读取server的信息，比如：
            1. list_tools: 列出server上所有的tool
            2. list_prompts: 列出server上所有的prompt
            3. list_resources: 列出server上所有的resource
            session可以用来调用server
            1. call_tool: 调用server上的tool
            2. call_prompt: 调用server上的prompt
            3. call_resource: 调用server上的resource
            """
            # List available tools
            tools_result = await session.list_tools()
            print("Available tools:")
            for tool in tools_result.tools:
                print(f"  - {tool.name}: {tool.description}")

            # Call our calculator tool
            result = await session.call_tool("func_add", arguments={"a": 2, "b": 3})
            print(f"2 + 3 = {result.content[0].text}")


if __name__ == "__main__":
    asyncio.run(main())