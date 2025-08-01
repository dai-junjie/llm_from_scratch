主要学习RAG，agent，multi-agents，微调技术
## RAG
RAG的实现步骤包括
● Data Ingestion（数据采集）: 加载和预处理文本数据。
● Chunking（分块处理）: 将数据分割成更小的块以提高检索性能。
● Embedding Creation（嵌入创建）: 使用嵌入模型将文本块转换为数值表示。
● Semantic Search（语义搜索）: 根据用户查询检索相关块。
● Response Generation（响应生成）：使用语言模型根据检索到的文本生成响应。

1. simple RAG
直接对文本进行分块，但是有分块之间有一定的重叠度。计算query和所有分块的embedding，然后用query逐个与分块求cosine相似度，取出最相似的前k个。这前k个被当作context上下文，与用户问题一起送入LLM，然后生成回答。最后用LLM根据标准回答来判断生成的回答时候符合。
2. 语义分块 semantic chunking RAG
还是先把文本分为固定大小的分块，该方法的主要思想是语义相近的连续块最后被合并在一起。主要方法是先计算相邻块的cosine相似度，接下来计算第x百分位数作为分块阈值。百分位数是一个统计概念，第x百分位数表示有x%的数据值小于或等于该数值。在语义分块中，我们通过计算相似度数组的百分位数来确定合适的阈值，从而识别语义边界。
具体来说，如果我们选择第80百分位数作为阈值，那么80%的相似度值都会小于或等于这个阈值。实际的阈值大小取决于相似度数据的分布情况。
举例说明：
● 相似度数组: [0.2, 0.4, 0.6, 0.8, 0.9]
● 第80百分位数: 0.84
● 这意味着80%的相似度值（即0.2, 0.4, 0.6, 0.8）都小于或等于0.84
小于0.84的都被认为是相似度不够，那么对应位置的相邻块要分割，否则合并。
3. chunk size selector 选择分块大小
核心思想是，设定一个chunk size候选集[128,256,512]，对不同的chunk size做RAG查询，让LLM评估不同的chunk size下得到的结果，选择能得到最优结果的chunk size作为之后的chunk size。
4. 上下文增强检索 Contextual Enriched RAG
检索相关文本块时，不仅返回最相关的块，还包含其相邻的上下文块，提供更完整的信息背景。通过设置context_size参数控制包含的相邻块数量，确保检索到的信息具有更好的连贯性和完整性。

5. 块头部标题 Contextual Chunk Headers RAG  
为每个文本块使用LLM生成描述性标题，然后同时对文本内容和标题进行向量化。在检索时计算查询与文本内容和标题嵌入的平均相似度，提高检索精确度和语义匹配效果。

6. 文档增强RAG Document Augmentation RAG
核心创新是为每个文本块生成相关问题。具体步骤：
● 对每个文本块生成3-5个只能通过该块回答的问题
● 构建混合向量存储，同时包含原始文本块和生成的问题
● 检索时既匹配原文也匹配问题，通过问题-文档映射增强召回率
● 使用SimpleVectorStore实现向量存储和相似度搜索
## Agent
### langchain
[LangChain文档](https://python.langchain.com/docs/concepts/)
1. Chat models and prompts: Build a simple LLM application with prompt templates and chat models.单轮对话
2. Semantic search: Build a semantic search engine over a PDF with document loaders, embedding models, and vector stores.集成RAG
3. Classification: Classify text into categories or labels using chat models with structured outputs.结构化输出
### langgraph 
[langgraph文档](https://langchain-ai.github.io/langgraph/agents/agents/)
入门学习内容
1. 模拟调用天气tool的agent
2. 构建chatbot&使用prebuilts构建chatbot
3. 构建带有webtool的chatbot
4. 构建带有memory功能的chatbot
5. 构建带有人类assistance的chatbot
6. 自定义state
7. 回退到指定state
8. 构建一个langgraph server
9. workflows&agent 概念

项目实战
复现：[fullstack langgraph quickstart](https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart)

## Multi-Agents
### trae-agent
[trae-agent仓库](https://github.com/bytedance/trae-agent)
## 微调
