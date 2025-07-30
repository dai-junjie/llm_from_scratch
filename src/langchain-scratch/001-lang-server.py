from fastapi import FastAPI
from demo1 import get_chain
from langserve import add_routes

app = FastAPI(title="langchain-study",
              description="langchain的http接口",
              version="0.1.0",)

# 使用 langserve 的 add_routes 函数添加标准化的路由
chain = get_chain()
add_routes(app, chain, path="/chain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
    