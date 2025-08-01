"""
上下文增强检索

检索增强生成（RAG）通过从外部来源检索相关知识来增强 AI 的响应。传统的检索方法返回孤立的文本片段，这可能导致答案不完整。
为了解决这个问题，我们引入了上下文增强检索，确保检索到的信息包括相邻的片段，以实现更好的连贯性。

实现步骤：
- 数据采集：从 PDF 中提取文本
- 重叠上下文分块：将文本分割成重叠的块以保留上下文
- 嵌入创建：将文本块转换为数值表示
- 上下文感知检索：检索相关块及其邻居以获得更好的完整性
- 回答生成：使用语言模型根据检索到的上下文生成回答。
- 评估：使用评估数据集评估模型性能。
"""

import fitz
import os
import numpy as np
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY")
)

# ============================PDF中提取文本函数============================
def extract_text_from_pdf(pdf_path):
    """
    从 PDF 文件中提取文本，并打印前 `num_chars` 个字符。

    Args:
    pdf_path (str): Path to the PDF file.

    Returns:
    str: Extracted text from the PDF.
    """
    # 打开 PDF 文件
    mypdf = fitz.open(pdf_path)
    all_text = ""  # 初始化一个空字符串以存储提取的文本

    # 遍历 PDF 中的每一页
    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]
        text = page.get_text("text")  # 从页面中提取文本
        all_text += text  # 将提取的文本追加到 all_text 字符串中

    return all_text  # 返回提取的文本

# ============================文本分块============================
def chunk_text(text, n, overlap):
    """
    将文本分割为重叠的块

    Args:
    text (str): 要分割的文本
    n (int): 每个块的字符数
    overlap (int): 块之间的重叠字符数

    Returns:
    List[str]: 文本块列表
    """
    chunks = []  # 初始化文本块列表
    for i in range(0, len(text), n - overlap):
        # 添加从当前索引到索引 + 块大小的文本块
        chunks.append(text[i:i + n])

    return chunks  # 返回文本块列表


# 定义 PDF 文件的路径
pdf_path = "data/AI_Information.en.zh-CN.pdf"

# 从 PDF 文件中提取文本
extracted_text = extract_text_from_pdf(pdf_path)

# 将提取的文本分割成1000个字符的段落，重叠200个字符
text_chunks = chunk_text(extracted_text, 1000, 200)

# 打印创建的文本块数量
print("Number of text chunks:", len(text_chunks))

# 打印第一个文本块
print("\nFirst text chunk:")
print(text_chunks[0])


print('='*100)

# ============================创建词向量============================
def create_embeddings(texts):
    """
    为文本列表生成嵌入

    Args:
    texts (List[str]): 输入文本列表.

    Returns:
    List[np.ndarray]: List of numerical embeddings.
    """
    # 确保每次调用不超过64条文本
    batch_size = 10
    embeddings = []
    
    if type(texts) != list:
        texts = [texts]

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(
            model=os.getenv("EMBEDDING_MODEL_ID"),
            input=batch
        )
        # 将响应转换为numpy数组列表并添加到embeddings列表中
        embeddings.extend([np.array(embedding.embedding) for embedding in response.data])

    return embeddings

response = create_embeddings(text_chunks)

# ============================余弦相似度计算============================
def cosine_similarity(vec1, vec2):
    """
    计算两个向量之间的余弦相似度。

    Args:
    vec1 (np.ndarray): First vector.
    vec2 (np.ndarray): Second vector.

    Returns:
    float: Cosine similarity score.
    """

    # 计算两个向量的点积
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# ============================富上下文搜索============================
def context_enriched_search(query, text_chunks, embeddings, k=1, context_size=1):
    """
    检索最相关的文本块及其相邻的上下文块

    Args:
    query (str): 搜索查询
    text_chunks (List[str]): 文本块列表
    embeddings (List[dict]): 文本块嵌入列表
    k (int): 要检索的相关块数量
    context_size (int): 包含的相邻块数量

    Returns:
    List[str]: 包含上下文信息的相关文本块
    """
    # 将查询转换为嵌入向量
    query_embedding = create_embeddings(query)[0]  # 修复：获取第一个嵌入
    similarity_scores = []

    # 计算查询与每个文本块嵌入之间的相似度分数
    for i, chunk_embedding in enumerate(embeddings):
        # 计算查询嵌入与当前文本块嵌入之间的余弦相似度
        similarity_score = cosine_similarity(query_embedding, chunk_embedding)  # 修复：移除np.array包装
        # 将索引和相似度分数存储为元组
        similarity_scores.append((i, similarity_score))

    # 按相似度分数降序排序（相似度最高排在前面）
    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    # 获取最相关块的索引
    # top_index = [index for index, _ in similarity_scores[:k]]
    top_index = similarity_scores[0][0]

    # 定义上下文包含的范围
    # 确保不会超出 text_chunks 的边界
    start = max(0, top_index - context_size)
    end = min(len(text_chunks), top_index + context_size + 1)

    # 返回最相关的块及其相邻的上下文块
    return [text_chunks[i] for i in range(start, end)]

# ============================调用以上函数进行带上下文检索的查询============================
# 从 JSON 文件加载验证数据集
with open('data/val.json', encoding="utf-8") as f:
    data = json.load(f)

# 从数据集中提取第一个问题作为查询
query = data[0]['question']

# 检索最相关的块及其相邻的上下文块
# 参数说明:
# - query: 我们要搜索的问题
# - text_chunks: 从 PDF 中提取的文本块
# - response: 文本块的嵌入
# - k=1: 返回最佳匹配
# - context_size=1: 包括最佳匹配前后的各一个块作为上下文
top_chunks = context_enriched_search(query, text_chunks, response, k=1, context_size=1)

# 打印查询以供参考
print("Query:", query)
# 打印每个检索到的块，并附上标题和分隔符
for i, chunk in enumerate(top_chunks):
    print(f"Context {i + 1}:\n{chunk}\n=====================================")
    
    
# ============================经过检索得到了相关的片段，现在加上query来得到响应============================
# AI 助手的系统提示
system_prompt = "你是一个AI助手，严格根据给定的上下文进行回答。如果无法直接从提供的上下文中得出答案，请回复：'我没有足够的信息来回答这个问题。'"

def generate_response(system_prompt, user_prompt):
    """
    基于检索到的文本块生成 AI 回答。

    Args:
    system_prompt (str): 系统提示
    user_prompt (str): 用户提示

    Returns:
    str: AI-generated response.
    """
    # 使用指定的模型生成 AI 回答
    response = client.chat.completions.create(
        model=os.getenv("LLM_MODEL_ID"),
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # 返回 AI 回答的内容
    return response.choices[0].message.content

# 将检索到的文本块合并为一个上下文字符串
context = "\n".join([f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(top_chunks)])

# 通过组合上下文和查询创建用户提示
user_prompt = f"{context}\n\nQuestion: {query}"
ai_response = generate_response(system_prompt, user_prompt)
print("AI Response:\n", ai_response)

print('='*100)

# ============================评估响应的质量============================
evaluate_system_prompt = "你是一个智能评估系统，负责评估AI助手的回答。如果AI助手的回答与真实答案非常接近，则评分为1。如果回答错误或与真实答案不符，则评分为0。如果回答部分符合真实答案，则评分为0.5。"

evaluation_prompt = f"用户问题: {query}\nAI回答:\n{ai_response}\nTrue Response: {data[0]['ideal_answer']}\n{evaluate_system_prompt}"

# 使用评估系统提示和评估提示生成评估回答
evaluation_response = generate_response(evaluate_system_prompt, evaluation_prompt)
print(evaluation_response)