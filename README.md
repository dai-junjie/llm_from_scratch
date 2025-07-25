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
4. todo
## Agent
### langchain
[LangChain文档](https://python.langchain.com/docs/concepts/)
1. Chat models and prompts: Build a simple LLM application with prompt templates and chat models.单轮对话
2. Semantic search: Build a semantic search engine over a PDF with document loaders, embedding models, and vector stores.集成RAG
3. Classification: Classify text into categories or labels using chat models with structured outputs.结构化输出
### langgraph 
[langgraph文档](https://langchain-ai.github.io/langgraph/agents/agents/)
## Multi-Agents
### trae-agent
[trae-agent仓库](https://github.com/bytedance/trae-agent)
## 微调
