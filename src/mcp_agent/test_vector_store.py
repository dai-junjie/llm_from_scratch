from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore

from dotenv import load_dotenv
load_dotenv()

import os

query = "graph"
retriever = SKLearnVectorStore(
        embedding=DashScopeEmbeddings(
            model=os.getenv("EMBEDDING_MODEL_ID"),
            dashscope_api_key=os.getenv("LLM_API_KEY")
        ), 
        # persist_path=os.getcwd()+"/sklearn_vectorstore.parquet", 
        persist_path = os.getcwd()+"/src/mcp_agent/sklearn_vectorstore.parquet",
        serializer="parquet"
        ).as_retriever(search_kwargs={"k": 3})

# print('persist_path', os.getcwd()+"/src/mcp_agent/sklearn_vectorstore.parquet")
relevant_docs = retriever.invoke(query)

print(relevant_docs)


