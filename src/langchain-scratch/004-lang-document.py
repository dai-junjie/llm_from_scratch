from langchain_core.documents import Document
from langchain_chroma import Chroma
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import RunnableLambda
from langchain_openai import ChatOpenAI
from demo1 import get_embedding

# 模型model
model = ChatOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key = os.getenv("QWQ_API_KEY"),
    model="qwq-plus-latest",
    streaming=True,
    temperature=0.5,
    top_p = 0.5,
    )

documents =[
    Document(
        page_content="狗是伟大的伴侣，以其忠诚和友好而闻名",
        metadata={
            "source":"source1",
            "page":10,
        },
    ),
    Document(
        page_content="猫是优雅的动物，具有独立的性格和灵活的身手",
        metadata={
            "source": "source2",
            "page": 1
        }
    ),
    Document(
        page_content="鹦鹉是聪明的鸟类，能够模仿人类说话",
        metadata={
            "source": "source3",
            "page": 2
        }
    ),
    Document(
        page_content="兔子是温顺的宠物，适合家庭饲养",
        metadata={
            "source": "source4",
            "page": 3
        }
    ),
    Document(
        page_content="仓鼠虽然体型小巧，但非常活泼可爱",
        metadata={
            "source": "source5",
            "page": 4
        }
    ),
    Document(
        page_content="Hello, world!",
        metadata={"source": "https://example.com"}
    )
]

# 实例化一个向量空间
# 映射到向量空间，可以根据余弦相似度来计算相似度
vector_store = Chroma.from_documents(
    documents,
    embedding=get_embedding()
    )

query = "狗"
# vector_store.similarity_search(query)

# res = vector_store.similarity_search_with_score(query)

# print(res)

# 检索器
retriever = RunnableLambda(vector_store.similarity_search).bind(k=1) # 封装函数进去,选择相似度最高的一个

# res = retriever.batch(["狗","hello"])
# print(res)

# 提示词模版
message = """
使用提供的上下文来回答问题,{question},
上下文:{context}
"""
prompt_temp = ChatPromptTemplate.from_messages([
    ("human",message)
])

chain = {'question':RunnablePassthrough(),
         'context':retriever,
         } | prompt_temp | model

for chunk in chain.stream("狗是什么呢"):
    print(chunk.content,end="",flush=True)