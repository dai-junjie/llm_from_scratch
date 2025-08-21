import asyncio
import json
import os
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

import nest_asyncio
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI

# Apply nest_asyncio to allow nested event loops (needed for Jupyter/IPython)
nest_asyncio.apply()

# Load environment variables
load_dotenv()


class MCPOpenAIClient:
    """使用MCP工具与OpenAI模型交互的客户端。"""

    def __init__(self, model: str = "qwen-plus-latest"):
        """初始化OpenAI MCP客户端。

        参数:
            model: 要使用的OpenAI模型。
        """
        # 初始化会话和客户端对象
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.openai_client = AsyncOpenAI(
            base_url=os.getenv("LLM_BASE_URL"),
            api_key=os.getenv("LLM_API_KEY"),
        )
        self.model = model
        self.stdio: Optional[Any] = None
        self.write: Optional[Any] = None

    async def connect_to_server(self, server_script_path: str = "server.py"):
        """连接到MCP服务器。

        参数:
            server_script_path: 服务器脚本的路径。
        """
        # 服务器配置
        server_params = StdioServerParameters(
            command="python",
            args=[server_script_path],
        )

        # 连接到服务器
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        # 初始化会话
        await self.session.initialize()

        # 列出可用工具
        tools_result = await self.session.list_tools()
        print("\n已连接到服务器，可用工具:")
        for tool in tools_result.tools:
            print(f"  - {tool.name}: {tool.description}")

    async def get_mcp_tools(self) -> List[Dict[str, Any]]:
        """以OpenAI格式获取MCP服务器中的可用工具。

        返回:
            OpenAI格式的工具列表。
        """
        tools_result = await self.session.list_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in tools_result.tools
        ]

    async def process_query(self, query: str) -> str:
        """使用OpenAI和可用的MCP工具处理查询。

        参数:
            query: 用户查询。

        返回:
            来自OpenAI的响应。
        """
        # 获取可用工具
        tools = await self.get_mcp_tools()

        # 初始 OpenAI API 调用
        response = await self.openai_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": query}],
            tools=tools,
            tool_choice="auto",
        )

        # 获取助手的响应
        assistant_message = response.choices[0].message

        # 初始化会话，包含用户查询和助手响应
        messages = [
            {"role": "user", "content": query},
            assistant_message,
        ]

        # 处理工具调用
        if assistant_message.tool_calls:
            # 处理每个工具调用
            for tool_call in assistant_message.tool_calls:
                # 执行工具调用
                result = await self.session.call_tool(
                    tool_call.function.name,
                    arguments=json.loads(tool_call.function.arguments),
                )

                # 添加工具响应到会话
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result.content[0].text,
                    }
                )

            # 获取包含工具结果的最终响应
            final_response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="none",  # 不允许更多工具调用
            )

            return final_response.choices[0].message.content

        # 没有工具调用，直接返回助手响应
        return assistant_message.content

    async def cleanup(self):
        """清理资源。"""
        await self.exit_stack.aclose()


async def main():
    """客户端主入口。"""
    client = MCPOpenAIClient()
    await client.connect_to_server("src/mcp_agent/openai-integration-mcp-server.py")

    # 示例：询问公司的休假政策
    query = "我们公司的休假政策是什么？"
    print(f"\n问题: {query}")

    response = await client.process_query(query)
    print(f"\n回答: {response}")


if __name__ == "__main__":
    asyncio.run(main())
