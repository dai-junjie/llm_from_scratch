本目录是在学习langgraph官方文档后，复习使用的。学习过程代码敲的少，大多是复制然后简单修改一下官方教程的代码。

> 重新对一些知识点进行了认识：state，graph，node，edge，conditional_edge,tool,Annotation,bind_tools,message,add_messages,node函数返回state的一些操作,条件边函数中的路径映射。多轮对话记忆管理的不同方式(手动，自动也就是利用节点函数返回state自动更新旧的state)

本目录下y开头的py文件学习自：[langgraph tutorial](https://www.youtube.com/watch?v=UklCxmEvz2w)

本目录下o开头的py文件学习子:[langgraph官方教程](https://www.youtube.com/watch?v=aHCDrAbH_go)
教学文档在[tutorial](https://mirror-feeling-d80.notion.site/Workflow-And-Agents-17e808527b1780d792a0d934ce62bee6)
官方参考文档在[documentation](https://langchain-ai.github.io/langgraph/tutorials/workflows/)


## LangGraph 示例代码总结

### o系列示例（官方教程）

#### o1_joke_agent.ipynb

这个示例展示了如何构建一个简单的笑话生成代理。主要功能包括：

- 定义了一个包含`topic`、`joke`、`improved_joke`和`final_joke`字段的`State`类型
- 使用`ChatOpenAI`模型作为基础LLM
- 实现了三个核心函数：
  - `generate_joke`：根据主题生成初始笑话
  - `improve_joke`：改进初始笑话
  - `polish_joke`：对笑话进行最终润色
- 使用`check_punchline`函数检查笑话是否包含标点符号
- 通过`StateGraph`将这些函数组织成一个工作流，形成一个完整的笑话生成代理

#### o2_parallelized_sub_tasks.ipynb

这个示例展示了如何实现并行化处理子任务。主要特点：

- 定义了包含`topic`、`joke`、`story`、`poem`和`combined_output`字段的`State`类型
- 实现了三个并行执行的LLM调用函数：
  - `call_llm_1`：生成关于主题的笑话
  - `call_llm_2`：生成关于主题的故事
  - `call_llm_3`：生成关于主题的诗歌
- 使用`aggregator`函数将三个并行任务的结果合并成一个输出
- 通过`StateGraph`构建工作流，从`START`节点同时连接到三个LLM调用节点，然后将它们的输出连接到`aggregator`节点

#### o3_routing.ipynb

这个示例可能展示了如何在LangGraph中实现路由功能，即根据不同条件将执行流程导向不同的节点。

#### o4_orchestrator.ipynb

这个示例展示了如何构建一个编排器（orchestrator）来生成结构化报告。主要功能包括：

- 使用Pydantic模型定义结构化输出：
  - `Section`类：包含`name`和`description`字段
  - `Sections`类：包含`sections`列表
- 定义了包含`topic`、`sections`、`completed_sections`和`final_report`字段的`State`类型
- 实现了三个主要函数：
  - `orchestrator`：生成报告计划，确定报告的各个部分
  - `llm_call`：为报告的每个部分生成内容
  - `synthesizer`：将所有部分合并成最终报告
- 使用`assign_workers`函数为报告的每个部分分配工作者
- 通过`StateGraph`构建工作流，实现从计划生成到内容创建再到最终合成的完整流程

#### o5_evaluator_optimizer.ipynb

这个示例展示了如何构建一个评估器-优化器工作流，用于生成和改进笑话。主要功能包括：

- 使用Pydantic模型定义结构化输出：
  - `FeedBack`类：包含`grade`（'funny'或'not funny'）和`feedback`字段
- 定义了包含`joke`、`topic`、`feedback`和`funny_or_not`字段的`State`类型
- 实现了两个核心函数：
  - `llm_call_generator`：根据主题和反馈生成笑话
  - `llm_call_evaluator`：评估笑话是否有趣并提供改进建议
- 使用`route_joke`条件边函数根据评估结果决定是接受笑话还是返回生成器进行改进
- 通过`StateGraph`构建循环优化工作流，实现笑话的迭代改进直到达到满意标准

#### o6_beyound_workflow.ipynb

这个示例展示了如何构建一个超越简单工作流的智能代理，能够自主决定何时使用工具。主要功能包括：

- 定义了三个数学运算工具：`add`、`multiply`、`divide`
- 使用`MessagesState`作为状态类型，管理对话消息流
- 实现了两个核心节点：
  - `llm_call`：LLM决定是否调用工具以及调用哪个工具
  - `tool_node`：执行工具调用并返回结果
- 使用`should_continue`条件边函数根据LLM是否发出工具调用来决定下一步流程
- 通过`StateGraph`构建循环代理，实现LLM自主决策和工具使用的智能工作流
- 展示了如何通过简单的system message和历史消息，让LLM能够分步解决复杂问题

### y系列示例（YouTube教程）

#### y1_build_my_first_agent.py

这个示例展示了如何构建一个基本的LangGraph代理：

- 使用`TypedDict`定义`AgentState`类型，包含`user_message`字段
- 使用`add_messages`注解实现消息累加
- 创建一个简单的`first_node`函数处理用户消息并返回AI响应
- 构建`StateGraph`，添加节点和边，实现从START到END的流程
- 使用`InMemorySaver`实现对话记忆管理
- 实现一个交互式循环，允许用户输入消息并获取AI响应

#### y2_agent_with_context_feeding.py

这个示例展示了如何在LangGraph中实现上下文传递：

- 使用`TypedDict`定义`State`类型，包含`messages`字段存储对话历史
- 创建`our_processing_node`函数处理用户消息并更新对话历史
- 使用`SystemMessage`设置AI助手的角色（海盗风格）
- 实现消息流式处理，展示如何在每次交互中更新和维护对话历史

#### y3_agent_with_inbuilt_tools.py

这个示例展示了如何在LangGraph中使用内置工具和条件边：

- 使用`Annotated`类型注解和`BaseMessage`类型
- 创建自定义消息类型`DickMessage`
- 使用`TavilySearch`和自定义`news_tool`作为工具
- 实现`should_continue`条件边函数，根据消息是否包含工具调用决定下一步流程
- 使用`ToolNode`处理工具调用
- 构建带有条件边的`StateGraph`，实现工具调用和对话的流程

#### y4_ReAct_agent_easy_method.py

这个示例展示了如何使用LangGraph的预构建ReAct代理：

- 使用`create_react_agent`快速创建一个具有推理和行动能力的代理
- 集成`TavilySearch`工具
- 简化代理创建过程，无需手动定义状态和节点
- 展示如何流式处理代理响应

#### y5_our_own_tools.py

这个示例展示了如何创建和使用自定义工具：

- 使用`@tool`装饰器定义两个自定义工具：
  - `weather_tool`：获取城市天气信息
  - `social_media_follower_count`：获取社交媒体关注者数量
- 使用详细的文档字符串描述工具功能、参数和返回值
- 将自定义工具与`create_react_agent`集成
- 展示如何在一个查询中使用多个工具

这些示例展示了LangGraph框架的不同功能和用例，包括基本代理构建、上下文管理、工具集成、条件路由、预构建代理使用和自定义工具创建，为构建复杂的LLM应用提供了参考实现。

