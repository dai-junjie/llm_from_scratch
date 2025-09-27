#!/usr/bin/env python3
"""
MCP STDIO Transport 测试脚本
基于main.py的结构，使用MCP Python SDK测试服务器的STDIO传输功能
"""

import asyncio
import nest_asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

nest_asyncio.apply()  # Needed to run interactive python


async def test_mcp_stdio_server():
    """使用MCP SDK测试STDIO服务器"""
    result = {
        'success': False,
        'message': '',
        'tools': [],
        'server_info': None,
        'error': None
    }
    
    # 配置参数 - 参考main.py的配置风格
    server_params = StdioServerParameters(
        command="python",
        args=["/Users/daijunjie/code/mcp/deploy-test-mcp/server.py"],
        transport="stdio"
    )
    
    try:
        print("启动MCP STDIO服务器测试...")
        
        # 创建客户端会话来与服务器进行交互
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                print("初始化连接...")
                
                # 初始化连接
                await session.initialize()
                print("连接初始化成功")
                
                print("获取可用工具列表...")
                
                # 列出可用的工具  
                tools_result = await session.list_tools()
                result['tools'] = [
                    {
                        'name': tool.name,
                        'description': tool.description
                    }
                    for tool in tools_result.tools
                ]
                
                result['success'] = True
                result['message'] = "MCP STDIO服务器连接成功"
                
                print(f"找到 {len(result['tools'])} 个可用工具:")
                for tool in result['tools']:
                    print(f"  - {tool['name']}: {tool['description']}")
                    
    except Exception as e:
        result['error'] = str(e)
        result['message'] = f"测试过程中发生错误: {str(e)}"
    
    return result


async def test_deploy_tool():
    """测试部署工具功能"""
    result = {
        'success': False,
        'message': '',
        'tool_response': None,
        'error': None
    }
    
    # 配置参数 - 参考main.py的配置风格  
    server_params = StdioServerParameters(
        command="python",
        args=["/Users/daijunjie/code/mcp/deploy-test-mcp/server.py"],
        transport="stdio"
    )
    
    # 测试项目路径 - 参考main.py中的project_path配置
    test_project_path = '/Users/daijunjie/code/mcp/deploy-test-mcp/projects'
    
    try:
        print("测试部署工具功能...")
        print("注意：如果端口被占用，部署可能会失败但工具调用仍然成功")
        
        # 创建客户端会话
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                # 初始化连接
                await session.initialize()
                
                print(f"调用deploy_test工具，项目路径: {test_project_path}")
                
                # 调用部署工具 - 参考main.py中deploy_docker_service的调用方式
                tool_result = await session.call_tool(
                    "deploy_test", 
                    arguments={"project_path": test_project_path}
                )
                
                # 解析工具响应
                if tool_result.content:
                    if hasattr(tool_result.content[0], 'text'):
                        # 文本响应
                        response_text = tool_result.content[0].text
                        try:
                            result['tool_response'] = json.loads(response_text)
                        except json.JSONDecodeError:
                            result['tool_response'] = response_text
                    else:
                        # 其他类型响应
                        result['tool_response'] = str(tool_result.content[0])
                        
                    result['success'] = True
                    result['message'] = "部署工具调用成功"
                else:
                    result['message'] = "工具调用未返回内容"
                    
    except Exception as e:
        result['error'] = str(e)
        result['message'] = f"工具测试过程中发生错误: {str(e)}"
    
    return result


async def main():
    """主测试函数 - 参考main.py的结构"""
    print("=== MCP STDIO 传输测试 ===\n")
    
    # 测试服务器初始化
    init_result = await test_mcp_stdio_server()
    
    # 打印初始化结果 - 参考main.py的输出格式
    print("\n=== 服务器初始化测试结果 ===")
    print(f"测试成功: {init_result['success']}")
    print(f"消息: {init_result['message']}")
    
    if init_result['error']:
        print(f"错误: {init_result['error']}")
    
    print("\n" + "="*50 + "\n")
    
    # 测试部署工具
    tool_result = await test_deploy_tool()
    
    # 打印部署结果 - 参考main.py的输出格式
    print("=== 部署工具测试结果 ===")
    print(f"测试成功: {tool_result['success']}")
    print(f"消息: {tool_result['message']}")
    
    if tool_result['tool_response']:
        print("工具响应:")
        if isinstance(tool_result['tool_response'], dict):
            # 参考main.py中的结果输出格式
            response = tool_result['tool_response']
            if 'success' in response:
                print(f"  部署成功: {response['success']}")
            if 'message' in response:
                print(f"  消息: {response['message']}")
            if 'data' in response:
                data = response['data']
                if 'container_id' in data:
                    print(f"  容器ID: {data['container_id']}")
                if 'access_url' in data:
                    print(f"  访问地址: {data['access_url']}")
        else:
            print(f"  {tool_result['tool_response']}")
    
    if tool_result['error']:
        print(f"错误: {tool_result['error']}")


if __name__ == "__main__":
    asyncio.run(main())