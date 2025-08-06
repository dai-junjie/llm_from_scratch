import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import DashScopeEmbeddings

def get_chain():
    os.environ['http_proxy'] = '127.0.0.1:7890'
    os.environ['https_proxy'] = '127.0.0.1:7890'
    os.environ['LANGCHAIN_TRACING_V2'] = 'false'

    # os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
    # os.environ["QWQ_API_KEY"] = "sk-304864592a0d47fc9c64b9093f1b559a"
    # 2.model
    model = get_model()
    # 3.prompt

    prompt = ChatPromptTemplate.from_messages([
        ('system', '你是一个{role}'),
        ('user', '喝{drink}有什么好处'),
    ])


    # 3. chain
    parser = StrOutputParser()

    chain = prompt | model | parser
    return chain

def get_model():
    model = ChatOpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key = os.getenv("QWQ_API_KEY"),
        model="qwq-plus-latest",
        streaming=True,
        temperature=0.5,
        top_p = 0.5,
        )
    return model

def get_embedding():
    embedding = DashScopeEmbeddings(dashscope_api_key=os.getenv("QWQ_API_KEY"))
    return embedding

# 写入markdown文件
# with open("output.md", "w") as f:
#     for chunk in chain.stream({"role":"计算机专业的硕士生导师","drink":"奶茶"}):
#         print(chunk, end="", flush=True)
#         f.write(chunk)
#         f.flush()
        
