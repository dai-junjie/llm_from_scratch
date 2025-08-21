from mcp.server.fastmcp import FastMCP
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore
from dotenv import load_dotenv
load_dotenv()

import os

mcp = FastMCP('langgraph-doc-mcp-server')
PATH = "src/mcp_agent/"

@mcp.tool()
def langgraph_query_tool(query: str):
    """
    Query the LangGraph documentation using a retriever.
    
    Args:
        query (str): The query to search the documentation with

    Returns:
        str: A str of the retrieved documents
    """
    persist_path = os.getcwd()+"/src/mcp_agent/sklearn_vectorstore.parquet"
    
    retriever = SKLearnVectorStore(
        embedding=DashScopeEmbeddings(
            model=os.getenv("EMBEDDING_MODEL_ID"),
            dashscope_api_key=os.getenv("LLM_API_KEY")
        ), 
        persist_path=persist_path, 
        serializer="parquet"
        ).as_retriever(search_kwargs={"k": 3})

    relevant_docs = retriever.invoke(query)
    print(f"Retrieved {len(relevant_docs)} relevant documents")
    formatted_context = "\n\n".join([f"==DOCUMENT {i+1}==\n{doc.page_content}" for i, doc in enumerate(relevant_docs)])
    return formatted_context

# The @mcp.resource() decorator is meant to map a URI pattern to a function that provides the resource content
@mcp.resource("docs://langgraph/full")
def get_all_langgraph_docs() -> str:
    """
    Get all the LangGraph documentation. Returns the contents of the file llms_full.txt,
    which contains a curated set of LangGraph documentation (~300k tokens). This is useful
    for a comprehensive response to questions about LangGraph.

    Args: None

    Returns:
        str: The contents of the LangGraph documentation
    """

    # Local path to the LangGraph documentation
    doc_path = PATH + "llms_full.txt"
    print('try to read file from', doc_path)
    try:
        with open(doc_path, 'r') as file:
            return file.read()
    except Exception as e:
        return f"Error reading log file: {str(e)}"

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')