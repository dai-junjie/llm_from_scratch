import fitz
import os
import numpy as np
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def extract_text_from_pdf(pdf_path):
    """
    从PDF文件中提取文本。

    参数：
    pdf_path (str): PDF文件路径。

    返回：
    str: 提取的文本内容。
    """
    # 打开PDF文件
    mypdf = fitz.open(pdf_path)
    all_text = ""  # 初始化一个空字符串用于存储提取的文本

    # 遍历PDF中的每一页
    for page in mypdf:
        # 提取当前页的文本并添加空格
        all_text += page.get_text("text") + " "

    # 返回去除首尾空白的提取文本
    return all_text.strip()

# 定义PDF文件路径
pdf_path = "./data/AI_Information.en.zh-CN.pdf"

# 从PDF文件中提取文本
extracted_text = extract_text_from_pdf(pdf_path)

# 打印提取文本的前500个字符
print(extracted_text[:500])

# 国内支持类OpenAI的API都可，需要配置对应的base_url和api_key
client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY")
)

def get_embedding(text):
    response = client.embeddings.create(
        model=os.getenv("EMBEDDING_MODEL_ID"),
        input=text
    )
    return np.array(response.data[0].embedding)

# 按句号分割文本为句子（基础分割）
sentences = extracted_text.split("。")
print(len(sentences))
# 为每个句子生成嵌入向量
embeddings = [get_embedding(sentence) for sentence in sentences if sentence]
print(f"Generated {len(embeddings)} sentence embeddings.")

# 计算连续句子的余弦相似度
def cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度。

    参数：
    vec1 (np.ndarray): 第一个向量。
    vec2 (np.ndarray): 第二个向量。

    返回：
    float: 余弦相似度。
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# 计算相邻句子的相似度
similarities = [cosine_similarity(embeddings[i], embeddings[i + 1]) for i in range(len(embeddings) - 1)]

# 实现了三种不同的方法来查找断点
def compute_breakpoints(similarities, method="percentile", threshold=90):
    """
    根据相似度下降计算分块的断点。

    参数：
        similarities (List[float]): 句子之间的相似度分数列表。
        method (str): 'percentile'（百分位）、'standard_deviation'（标准差）或 'interquartile'（四分位距）。
        threshold (float): 阈值（对于 'percentile' 是百分位数，对于 'standard_deviation' 是标准差倍数）。

    返回：
        List[int]: 分块的索引列表。
    """
    # 根据选定的方法确定阈值
    if method == "percentile":
        # 计算相似度分数的第 X 百分位数
        threshold_value = np.percentile(similarities, threshold)
    elif method == "standard_deviation":
        # 计算相似度分数的均值和标准差。
        mean = np.mean(similarities)
        std_dev = np.std(similarities)
        # 将阈值设置为均值减去 X 倍的标准差
        threshold_value = mean - (threshold * std_dev)
    elif method == "interquartile":
        # 计算第一和第三四分位数（Q1 和 Q3）。
        q1, q3 = np.percentile(similarities, [25, 75])
        # 使用 IQR 规则（四分位距规则）设置阈值
        threshold_value = q1 - 1.5 * (q3 - q1)
    else:
        # 如果提供了无效的方法，则抛出异常
        raise ValueError("Invalid method. Choose 'percentile', 'standard_deviation', or 'interquartile'.")

    # 找出相似度低于阈值的索引
    return [i for i, sim in enumerate(similarities) if sim < threshold_value]

# 使用百分位法计算断点，阈值为90
breakpoints = compute_breakpoints(similarities, method="percentile", threshold=90)
print(f'breakpoints: {breakpoints}')

# 文本基于断点进行分割
def split_into_chunks(sentences, breakpoints):
    """
    将句子分割为语义块

    参数：
    sentences (List[str]): 句子列表
    breakpoints (List[int]): 进行分块的索引位置

    返回：
    List[str]: 文本块列表
    """
    chunks = []  # 初始化一个空列表用于存储文本块
    start = 0  # 初始化起始索引

    # 遍历每个断点以创建块
    for bp in breakpoints:
        # 将从起始位置到当前断点的句子块追加到列表中
        chunks.append("。".join(sentences[start:bp + 1]) + "。")
        start = bp + 1  # 将起始索引更新为断点后的下一个句子

    # 将剩余的句子作为最后一个块追加
    chunks.append("。".join(sentences[start:]))
    return chunks  # 返回文本块列表

# split_into_chunks 函数创建文本块
text_chunks = split_into_chunks(sentences, breakpoints)

# 打印创建的块数量
print(f"Number of semantic chunks: {len(text_chunks)}")

# 打印第一个块以验证结果
print("\nFirst text chunk:")
print(text_chunks[0])

# 为每个片段创建嵌入，以便后续检索
def create_embeddings(text_chunks):
    """
    为每个文本块创建嵌入向量。

    参数：
    text_chunks (List[str]): 文本块列表。

    返回：
    List[np.ndarray]: 嵌入向量列表。
    """
    # 使用get_embedding函数为每个文本块生成嵌入向量
    return [get_embedding(chunk) for chunk in text_chunks]

# 使用create_embeddings函数创建块嵌入向量
chunk_embeddings = create_embeddings(text_chunks)

# 基于余弦相似度来检索最相关的片段
def semantic_search(query, text_chunks, chunk_embeddings, k=5):
    """
    查询找到最相关的文本块

    参数：
    query (str): 查询语句。
    text_chunks (List[str]): 文本块列表。
    chunk_embeddings (List[np.ndarray]): 块嵌入向量列表。
    k (int): 返回最相关结果的数量。

    返回：
    List[str]: 最相关的k个文本块。
    """
    # 为查询生成嵌入向量
    query_embedding = get_embedding(query)

    # 计算查询嵌入与每个块嵌入之间的余弦相似度
    similarities = [cosine_similarity(query_embedding, emb) for emb in chunk_embeddings]

    # 获取最相似的 k 个块的索引
    top_indices = np.argsort(similarities)[-k:][::-1]

    # 返回最相关的 k 个文本块
    return [text_chunks[i] for i in top_indices]

# 加载测试用的问答数据
with open('./data/val.json', encoding="utf-8") as f:
    data = json.load(f)

# 提取验证数据中的第一个查询
query = data[0]['question']

# 获取最相关的前2个文本块
top_chunks = semantic_search(query, text_chunks, chunk_embeddings, k=2)

# 打印查询
print(f"Query: {query}")

# 打印最相关的前2个文本块
for i, chunk in enumerate(top_chunks):
    print(f"Context {i+1}:\n{chunk}\n{'='*40}")
    
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
            model=os.getenv("LLM_MODEL_ID"),
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
user_prompt = "\n".join([f"上下文内容 {i + 1}:\n{chunk}\n=====================================\n" for i, chunk in enumerate(top_chunks)])
user_prompt = f"{user_prompt}\n问题: {query}"

# 生成AI回复
ai_response = generate_response(system_prompt, user_prompt)
print(f'\n\nai回答:{ai_response} \n', '='*100)

# 评估人工智能响应
# 定义评估系统的系统提示词
evaluate_system_prompt = "你是一个智能评估系统，负责评估AI助手的回答。如果AI助手的回答与真实答案非常接近，则评分为1。如果回答错误或与真实答案不符，则评分为0。如果回答部分符合真实答案，则评分为0.5。"

# 组合用户查询、AI回复、真实答案和评估系统提示词，生成评估提示词
evaluation_prompt = f"用户问题: {query}\nAI回答:\n{ai_response}\nTrue Response: {data[0]['ideal_answer']}\n{evaluate_system_prompt}"

# 使用评估系统提示词和评估提示词生成评估回复
evaluation_response = generate_response(evaluate_system_prompt, evaluation_prompt)
print(evaluation_response)