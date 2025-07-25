import fitz
import os
import numpy as np
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def extract_text_from_pdf(pdf_path):
    # 打开PDF文件
    mypdf = fitz.open(pdf_path)
    all_text = ""  # 初始化一个空字符串用于存储提取的文本

    # 遍历PDF中的每一页
    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]  # 获取当前页
        text = page.get_text("text")  # 从当前页提取文本
        all_text += text  # 将提取的文本追加到all_text字符串

    return all_text  # 返回提取的文本


def chunk_text(text, n, overlap):
    """
    将给定文本按n个字符为一块进行切分，并设置重叠度。

    参数：
    text (str): 文本
    n (int): 块长度
    overlap (int): 重叠度

    返回：
    List[str]: 文本块列表。
    """
    chunks = []  # 初始化一个空列表用于存储文本块

    # 以(n - overlap)为步长遍历文本
    for i in range(0, len(text), n - overlap):
        # 从索引i到i+n切分文本，并添加到chunks列表
        chunks.append(text[i:i + n])

    return chunks

# 国内支持类OpenAI的API都可，我用的是火山引擎的，需要配置对应的base_url和api_key


client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("QWQ_API_KEY"),
    # model='qwen-plus',
)

# PDF文件路径
pdf_path = "./data/AI_Information.en.zh-CN.pdf"

# 提取文本
extracted_text = extract_text_from_pdf(pdf_path)

# 切分文本块，块长度为500，重叠度为100
text_chunks = chunk_text(extracted_text, 500, 100)

# 文本块的数量
print("Number of text chunks:", len(text_chunks))

# 第一个文本块
print("\nFirst text chunk:")
print(text_chunks[0])


def create_embeddings(text):
    # 使用指定模型为输入文本创建嵌入向量

    # 如果是单个文本，直接处理
    if isinstance(text, str):
        response = client.embeddings.create(
            model="text-embedding-v4",
            input=text
        )
        return response

    # 如果是文本列表，分批处理
    batch_size = 10  # 阿里云限制
    all_embeddings = []
    for i in range(0, len(text), batch_size):
        batch = text[i:i + batch_size]
        response = client.embeddings.create(
            model="text-embedding-v4",
            input=batch
        )
        all_embeddings.extend(response.data)

    # 创建一个类似原始响应的对象
    class EmbeddingResponse:
        def __init__(self, data):
            self.data = data

    return EmbeddingResponse(all_embeddings)


# 文本块的嵌入向量
response = create_embeddings(text_chunks)

# 实现余弦相似度来找到与用户查询最相关的文本片段


def cosine_similarity(vec1, vec2):
    """
    计算两个向量之间的余弦相似度。

    参数：
    vec1 (np.ndarray): 第一个向量。
    vec2 (np.ndarray): 第二个向量。

    返回：
    float: 两个向量的余弦相似度。
    """
    # 计算两个向量的点积并除以它们范数的乘积
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def semantic_search(query, text_chunks, embeddings, k=5):
    """
    使用给定的查询和嵌入向量对文本块进行语义检索。

    参数：
    query (str): 查询语句。
    text_chunks (List[str]): 待检索的文本块列表。
    embeddings (List[dict]): 文本块的嵌入向量列表。
    k (int): 返回最相关文本块的数量，默认为5。

    返回：
    List[str]: 前k个最相关的文本块。
    """
    # 为查询创建嵌入向量
    query_embedding = create_embeddings(query).data[0].embedding
    similarity_scores = []  # 初始化相似度分数列表

    # 计算查询嵌入与每个文本块嵌入的相似度分数
    for i, chunk_embedding in enumerate(embeddings):
        similarity_score = cosine_similarity(
            np.array(query_embedding), np.array(chunk_embedding.embedding))
        similarity_scores.append((i, similarity_score))  # 添加索引和相似度分数

    # 按相似度分数降序排序
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    # 获取前k个最相似文本块的索引
    top_indices = [index for index, _ in similarity_scores[:k]]
    # 返回前k个最相关的文本块
    return [text_chunks[index] for index in top_indices]


# 从JSON文件加载验证数据
with open('./data/val.json', encoding="utf-8") as f:
    data = json.load(f)

# 提取验证数据中的第一个查询
query = data[0]['question']

# 语义检索，找到与查询最相关的前2个文本块
top_chunks = semantic_search(query, text_chunks, response.data, k=2)

# 打印查询
print("Query:", query)

# 打印最相关的前2个文本块
for i, chunk in enumerate(top_chunks):
    print(f"Context {i + 1}:\n{chunk}\n=====================================")


# 定义AI助手的系统提示词
system_prompt = "你是一个AI助手，严格根据给定的上下文进行回答。如果无法直接从提供的上下文中得出答案，请回复：'我没有足够的信息来回答这个问题。'"


def generate_response(system_prompt, user_message):
    """
    基于系统提示词和用户消息生成AI模型的回复。

    参数：
    system_prompt (str): 指导AI行为的系统提示词。
    user_message (str): 用户消息或查询。

    返回：
    dict: AI模型的回复。
    """
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.1,
        top_p=0.8,
        presence_penalty=1.05,
        max_tokens=4096,
    )
    return response.choices[0].message.content


# 基于top_chunks创建用户提示词
user_prompt = "\n".join(
    [f"上下文内容 {i + 1}:\n{chunk}\n=====================================\n" for i, chunk in enumerate(top_chunks)])
user_prompt = f"{user_prompt}\n问题: {query}"

# 生成AI回复
ai_response = generate_response(system_prompt, user_prompt)
print(f'\n\nai回答:{ai_response} \n', '='*100)

# 定义评估系统的系统提示词
evaluate_system_prompt = "你是一个智能评估系统，负责评估AI助手的回答。如果AI助手的回答与真实答案非常接近，则评分为1。如果回答错误或与真实答案不符，则评分为0。如果回答部分符合真实答案，则评分为0.5。"

# 组合用户查询、AI回复、真实答案和评估系统提示词，生成评估提示词
evaluation_prompt = f"用户问题: {query}\nAI回答:\n{ai_response}\nTrue Response: {data[0]['ideal_answer']}\n{evaluate_system_prompt}"

# 使用评估系统提示词和评估提示词生成评估回复
evaluation_response = generate_response(
    evaluate_system_prompt, evaluation_prompt)
print(evaluation_response)
