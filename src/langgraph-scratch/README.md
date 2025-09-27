# LangGraph 学习示例

本目录下文件学习自：[langgraph tutorial](https://www.youtube.com/watch?v=UklCxmEvz2w)

## 示例代码说明

### c1_weather_agent.ipynb
这个示例展示了如何使用LangGraph的预构建功能创建一个简单的天气查询代理。主要特点：
- 使用`create_react_agent`快速创建ReAct代理
- 定义简单的`get_weather`工具函数
- 展示如何自定义代理提示词
- 演示如何使用`InMemorySaver`添加记忆功能，使代理能够记住之前的对话内容

### c2_build_chatbot.ipynb
这个示例展示了如何从头开始构建一个基本的聊天机器人。主要特点：
- 使用`StateGraph`定义聊天机器人的状态机结构
- 使用`TypedDict`和`Annotated[list, add_messages]`定义状态结构
- 添加节点、入口点和出口点
- 展示了如何编译和运行图

### c3_webtools_chatbot.py 和 c3_webtools_chatbot_with_prebuilts.py
这些示例展示了如何为聊天机器人添加网络搜索工具。主要特点：
- 集成`TavilySearch`工具进行网络搜索
- 使用`bind_tools`将工具绑定到LLM
- 实现`BasicToolNode`处理工具调用
- 使用条件边缘路由逻辑，根据LLM是否请求使用工具来决定执行路径

### c4_chatbot_with_memory.ipynb
这个示例展示了如何为聊天机器人添加持久化记忆功能。主要特点：
- 使用`InMemorySaver`创建内存检查点
- 在图编译时添加记忆功能
- 演示如何使用线程ID来管理多个对话

### c5_chatbot_add_human_in_the_loop.ipynb
这个示例展示了如何实现人机协作的工作流程。主要特点：
- 使用`interrupt`函数暂停执行并等待人类输入
- 创建`human_assistance`工具请求人类帮助
- 展示如何恢复中断的执行并继续处理

### c6_customize_state.ipynb
这个示例展示了如何自定义状态结构以存储额外信息。主要特点：
- 扩展`State`类型添加自定义字段（name和birthday）
- 在工具内部更新状态
- 使用`Command`从工具内部发出状态更新

### c7_rewind__time_travel.ipynb
这个示例展示了如何实现时间旅行功能，允许回退到之前的状态。主要特点：
- 使用`get_state_history`获取检查点历史
- 演示如何在之前的时间点恢复执行
- 展示如何管理和操作状态历史

### map_reduce.py
这个示例展示了Map-Reduce模式的简单实现：
- 实现简单的文本处理Map-Reduce流程
- 使用`map_step`函数处理单个文本并计算词频
- 使用`reduce_step`函数合并所有映射结果

### websearch_chatbot.py 和 test_websearch_chatbot.py
这些文件实现了一个完整的带有网络搜索功能的聊天机器人API：
- 使用FastAPI创建Web服务
- 实现流式响应
- 支持人机协作中断和恢复
- 包含完整的测试套件验证功能

## 知识点回顾

1. **状态管理**：使用`TypedDict`和`Annotated`定义状态结构，使用reducer函数（如`add_messages`）控制状态更新方式

2. **图构建**：使用`StateGraph`定义节点和边，使用`START`和`END`标记入口和出口点

3. **工具集成**：使用`bind_tools`将工具绑定到LLM，使用`ToolNode`处理工具调用

4. **条件路由**：使用条件边缘函数动态决定执行路径

5. **记忆与持久化**：使用`InMemorySaver`等检查点机制实现对话历史记忆

6. **人机协作**：使用`interrupt`和`Command`实现人机协作工作流

7. **时间旅行**：使用状态历史实现回退和重新执行功能