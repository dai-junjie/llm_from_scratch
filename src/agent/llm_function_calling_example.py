#!/usr/bin/env python3
"""
LLM函数调用示例
展示如何直接调用LLM并处理其响应，包括函数调用的完整流程
"""

import json
import os
import inspect
import re
from typing import List, Dict, Any, get_type_hints
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()


class FunctionCallingExample:
    def __init__(self, api_key: str = None, base_url: str = None):
        """初始化LLM客户端"""
        self.client = OpenAI(
            api_key=api_key or os.getenv("LLM_API_KEY"),
            base_url=base_url or os.getenv("LLM_BASE_URL"),
        )
        
        # 定义可用的工具函数
        self.tools = {
            "get_current_weather": self.get_current_weather,
            "get_stock_price": self.get_stock_price,
            "calculate_math": self.calculate_math
        }
        
    def get_current_weather(self, location: str, unit: str = "celsius") -> Dict[str, Any]:
        """获取指定位置的天气信息
        
        Args:
            location (str): 城市和国家，例如：北京，中国
            unit (str, optional): 温度单位，'celsius' 或 'fahrenheit'。默认为 'celsius'
            
        Returns:
            Dict[str, Any]: 包含天气信息的字典
        """
        # 模拟天气数据
        weather_data = {
            "location": location,
            "temperature": 22 if unit == "celsius" else 72,
            "unit": unit,
            "description": "晴朗",
            "humidity": 65,
            "wind_speed": 10
        }
        return weather_data
    
    def get_stock_price(self, symbol: str) -> Dict[str, Any]:
        """获取股票价格
        
        Args:
            symbol (str): 股票代码，例如：AAPL, GOOGL
            
        Returns:
            Dict[str, Any]: 包含股票价格信息的字典
        """
        # 模拟股票数据
        stock_data = {
            "symbol": symbol.upper(),
            "price": 150.25,
            "change": 2.5,
            "change_percent": 1.69,
            "timestamp": datetime.now().isoformat()
        }
        return stock_data
    
    def calculate_math(self, expression: str) -> Dict[str, Any]:
        """计算数学表达式
        
        Args:
            expression (str): 数学表达式，例如：2+2, (10*5)/2
            
        Returns:
            Dict[str, Any]: 包含计算结果或错误信息的字典
        """
        try:
            # 注意：在生产环境中应使用更安全的数学表达式解析器
            result = eval(expression)
            return {
                "expression": expression,
                "result": result
            }
        except Exception as e:
            return {
                "expression": expression,
                "error": str(e)
            }
    
    def _python_type_to_json_schema(self, py_type: type) -> Dict[str, Any]:
        """将Python类型转换为JSON Schema类型"""
        type_mapping = {
            str: {"type": "string"},
            int: {"type": "integer"},
            float: {"type": "number"},
            bool: {"type": "boolean"},
            list: {"type": "array"},
            dict: {"type": "object"},
        }
        
        # 处理可选参数 (Union[T, None] 或 Optional[T])
        if hasattr(py_type, '__origin__') and py_type.__origin__ is dict:
            return {"type": "object"}
        
        return type_mapping.get(py_type, {"type": "string"})  # 默认为string
    
    def _parse_docstring(self, docstring: str) -> Dict[str, str]:
        """解析函数文档字符串，提取参数描述"""
        if not docstring:
            return {}
        
        param_descriptions = {}
        # 查找 Args 部分
        args_match = re.search(r'Args:\s*(.*?)(?:\n\s*\n|Returns:|$)', docstring, re.DOTALL)
        if args_match:
            args_text = args_match.group(1)
            # 解析每个参数
            for line in args_text.strip().split('\n'):
                line = line.strip()
                if ':' in line and '(' in line:
                    param_part, desc_part = line.split(':', 1)
                    param_name = param_part.split('(')[0].strip()
                    param_descriptions[param_name] = desc_part.strip()
        
        return param_descriptions
    
    def get_tools_definitions(self) -> List[Dict]:
        """使用反射技术自动生成工具函数的定义，用于传递给LLM"""
        tools = []
        
        for name, func in self.tools.items():
            # 获取函数签名
            sig = inspect.signature(func)
            # 获取类型提示
            type_hints = get_type_hints(func)
            # 获取文档字符串
            docstring = inspect.getdoc(func)
            
            # 解析文档字符串中的参数描述
            param_descriptions = self._parse_docstring(docstring)
            
            # 构建参数schema
            properties = {}
            required = []
            
            for param_name, param in sig.parameters.items():
                # 跳过self参数
                if param_name == 'self':
                    continue
                
                param_info = {}
                
                # 获取参数类型
                if param_name in type_hints:
                    param_info.update(self._python_type_to_json_schema(type_hints[param_name]))
                else:
                    param_info["type"] = "string"  # 默认类型
                
                # 获取参数描述
                if param_name in param_descriptions:
                    param_info["description"] = param_descriptions[param_name]
                
                # 检查是否为必需参数
                if param.default == inspect.Parameter.empty:
                    required.append(param_name)
                
                properties[param_name] = param_info
            
            # 构建工具定义
            tool_def = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": docstring.split('\n')[0] if docstring else "",  # 第一行为函数描述
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                }
            }
            
            tools.append(tool_def)
        
        return tools
    
    def call_llm_with_functions(self, user_input: str) -> Dict[str, Any]:
        """调用LLM并处理函数调用"""
        messages = [
            {"role": "system", "content": "你是一个有用的助手，可以获取天气信息、股票价格和进行数学计算。"},
            {"role": "user", "content": user_input}
        ]
        
        # 第一次调用LLM，可能包含函数调用
        response = self.client.chat.completions.create(
            model="qwen-plus-latest",
            messages=messages,
            tools=self.get_tools_definitions(),
            tool_choice="auto"
        )
        
        response_message = response.choices[0].message
        messages.append(response_message)  # 添加LLM的响应到消息历史
        
        # 检查是否有工具调用
        if response_message.tool_calls:
            # 处理所有工具调用
            for tool_call in response_message.tool_calls:
                print(f'工具调用: {tool_call}')
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                # 调用相应的函数
                if function_name in self.tools:
                    function_response = self.tools[function_name](**function_args)
                    
                    # 将函数响应添加到消息历史
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps(function_response)
                    })
                
            # 第二次调用LLM，获取最终响应
            second_response = self.client.chat.completions.create(
                model="qwen-plus-latest",
                messages=messages
            )
            
            return {
                "final_response": second_response.choices[0].message.content,
                "tool_calls": response_message.tool_calls,
                "messages": messages
            }
        else:
            # 没有工具调用，直接返回响应
            return {
                "final_response": response_message.content,
                "tool_calls": None,
                "messages": messages
            }
    
    def run_example(self):
        """运行函数调用示例"""
        print("=== LLM函数调用示例 ===\n")
        
        # 示例1: 查询天气
        print("示例1: 查询天气")
        result = self.call_llm_with_functions("今天北京的天气怎么样？")
        print(f"最终响应: {result['final_response']}\n")
        
        # 示例2: 查询股票价格
        print("示例2: 查询股票价格")
        result = self.call_llm_with_functions("苹果公司的股票价格是多少？")
        print(f"最终响应: {result['final_response']}\n")
        
        # 示例3: 数学计算
        print("示例3: 数学计算")
        result = self.call_llm_with_functions("请计算(25*4+10)/2的结果")
        print(f"最终响应: {result['final_response']}\n")
        
        # 示例4: 普通对话
        print("示例4: 普通对话")
        result = self.call_llm_with_functions("你好，介绍一下你自己")
        print(f"最终响应: {result['final_response']}\n")


def main():
    # 注意：需要设置LLM_API_KEY和LLM_BASE_URL环境变量
    example = FunctionCallingExample()
    
    try:
        example.run_example()
    except Exception as e:
        print(f"运行示例时出错: {e}")
        print("请确保已设置LLM_API_KEY和LLM_BASE_URL环境变量")


if __name__ == "__main__":
    main()