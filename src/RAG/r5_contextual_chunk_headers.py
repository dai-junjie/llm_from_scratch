import fitz
import os
import json
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY")
)
llm_model = os.getenv("LLM_MODEL_ID")
embedding_model = os.getenv("EMBEDDING_MODEL_ID")

pdf_path = "data/AI_Information.en.zh-CN.pdf"

# 提取文本函数
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

# ============================为chunk文本生成标题的函数============================
def generate_chunk_header(chunk):
    """
    使用 LLM 为给定的文本块生成标题/页眉

    Args:
        chunk (str): 要总结为标题的文本块
        model (str): 用于生成标题的模型

    Returns:
        str: 生成的标题/页眉
    """
    # 定义系统提示
    system_prompt = "为给定的文本生成一个简洁且信息丰富的标题。"

    # 根据系统提示和文本块生成标题
    response = client.chat.completions.create(
        model=llm_model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chunk}
        ]
    )

    # 返回生成的标题/页眉，去除任何前导或尾随空格
    return response.choices[0].message.content.strip()

# ============================将文本有一定重叠度的分块，并为分块生成标题============================
def chunk_text_with_headers(text, n, overlap):
    """
    将文本分割为较小的片段，并生成标题。

    Args:
        text (str): 要分块的完整文本
        n (int): 每个块的字符数
        overlap (int): 块之间的重叠字符数

    Returns:
        List[dict]: 包含 'header' 和 'text' 键的字典列表
    """
    chunks = []

    # 按指定的块大小和重叠量遍历文本
    for i in range(0, len(text), n - overlap):
        chunk = text[i:i + n]
        header = generate_chunk_header(chunk)  # 使用 LLM 为块生成标题
        chunks.append({"header": header, "text": chunk})  # 将标题和块添加到列表中

    return chunks

# 调用以上函数，读取pdf文本，并分块然后添加标题，最后得到chunk列表
extracted_text = extract_text_from_pdf(pdf_path)

# 使用标题对提取的文本进行分块
# 我们使用1000个字符的块大小和200个字符的重叠
text_chunks = chunk_text_with_headers(extracted_text, 1000, 200)

# 打印带有生成标题的示例块
print("Sample Chunk:")
print("Header:", text_chunks[0]['header'])
print("Content:", text_chunks[0]['text'])

print('='*100)

# ============================文本转换为embedding的函数============================
def create_embeddings(texts):
    """
    为文本列表生成嵌入

    Args:
        texts (List[str]): 输入文本列表.

    Returns:
        List[np.ndarray]: List of numerical embeddings.
    """
    response = client.embeddings.create(
        model=embedding_model,
        input=texts
    )
    return response.data[0].embedding

embeddings = []  # 初始化一个空列表来存储嵌入

# 使用进度条遍历每个文本块
for chunk in tqdm(text_chunks, desc="Generating embeddings"):
    # 为块的文本创建嵌入
    text_embedding = create_embeddings(chunk["text"])
    # print(text_embedding.shape)
    # 为块的标题创建嵌入
    header_embedding = create_embeddings(chunk["header"])
    # 将块的标题、文本及其嵌入追加到列表中
    embeddings.append({"header": chunk["header"], "text": chunk["text"], "embedding": text_embedding,
                       "header_embedding": header_embedding})
    
# ============================得到词嵌入后就进行语义搜索，也即是相似度排序============================
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


def semantic_search(query, chunks, k=5):
    """
    根据查询搜索最相关的块

    Args:
    query (str): 用户查询
    chunks (List[dict]): 带有嵌入的文本块列表
    k (int): 返回的相关chunk数

    Returns:
    List[dict]: Top-k most relevant chunks.
    """
    query_embedding = create_embeddings(query)
    # print(query_embedding)
    # print(query_embedding.shape)

    similarities = []

    # 遍历每个块以计算相似度分数
    for chunk in chunks:
        # 计算查询嵌入与块文本嵌入之间的余弦相似度
        sim_text = cosine_similarity(np.array(query_embedding), np.array(chunk["embedding"]))
        # sim_text = cosine_similarity(query_embedding, chunk["embedding"])

        # 计算查询嵌入与块标题嵌入之间的余弦相似度
        sim_header = cosine_similarity(np.array(query_embedding), np.array(chunk["header_embedding"]))
        # sim_header = cosine_similarity(query_embedding, chunk["header_embedding"])
        # 计算平均相似度分数
        avg_similarity = (sim_text + sim_header) / 2
        # 将块及其平均相似度分数追加到列表中
        similarities.append((chunk, avg_similarity))

    # 根据相似度分数按降序对块进行排序
    similarities.sort(key=lambda x: x[1], reverse=True)
    # 返回最相关的前k个块
    return [x[0] for x in similarities[:k]]

# 加载验证数据
with open('data/val.json', encoding="utf-8") as f:
    data = json.load(f)

query = data[0]['question']

# 检索最相关的前2个文本块
top_chunks = semantic_search(query, embeddings, k=2)

# 打印结果
print("Query:", query)
for i, chunk in enumerate(top_chunks):
    print(f"Header {i+1}: {chunk['header']}")
    print(f"Content:\n{chunk['text']}\n")
    
print('='*100)
# =====================基于检索到的片段进行回答=======================
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
context = "\n".join([f"Context {i+1}:\n{chunk['text']}" for i, chunk in enumerate(top_chunks)])

# 通过组合上下文和查询创建用户提示
user_prompt = f"{context}\n\nQuestion: {query}"

print(f'user_prompt: {user_prompt}')
ai_response = generate_response(system_prompt, user_prompt)
print("AI Response:\n", ai_response)
print('='*100)
# ====================对回答进行评估======================
evaluate_system_prompt = "你是一个智能评估系统，负责评估AI助手的回答。如果AI助手的回答与真实答案非常接近，则评分为1。如果回答错误或与真实答案不符，则评分为0。如果回答部分符合真实答案，则评分为0.5。"

evaluation_prompt = f"用户问题: {query}\nAI回答:\n{ai_response}\nTrue Response: {data[0]['ideal_answer']}\n{evaluate_system_prompt}"

# 使用评估系统提示和评估提示生成评估回答
evaluation_response = generate_response(evaluate_system_prompt, evaluation_prompt)
print(evaluation_response)