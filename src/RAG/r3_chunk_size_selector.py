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

# 从 PDF 中提取文本
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

# 对提取的文本进行分块
# 为了提高检索效率，我们将提取的文本分割成不同大小的重叠块

def chunk_text(text, n, overlap):
    """
    将文本分割为重叠的块。

    参数：
    text (str): 要分割的文本
    n (int): 每个块的字符数
    overlap (int): 块之间的重叠字符数

    返回：
    List[str]: 文本块列表
    """
    chunks = []  #
    for i in range(0, len(text), n - overlap):
        # 添加从当前索引到索引 + 块大小的文本块
        chunks.append(text[i:i + n])

    return chunks  # 返回文本块列表

# 定义要评估的不同块大小
chunk_sizes = [128, 256, 512]

# 创建一个字典，用于存储每个块大小对应的文本块
text_chunks_dict = {size: chunk_text(extracted_text, size, size // 5) for size in chunk_sizes}

# 打印每个块大小生成的块数量
for size, chunks in text_chunks_dict.items():
    print(f"Chunk Size: {size}, Number of Chunks: {len(chunks)}")
    

# 嵌入将文本转换为数值表示，以进行相似性搜索。
from tqdm import tqdm
import numpy as np
import os
# 假设client已经被正确初始化和配置

def create_embeddings(texts):
    """
    为文本列表生成嵌入

    参数：
    texts (List[str]): 输入文本列表。

    返回：
    List[np.ndarray]: 数值嵌入列表。
    """
    # 确保每次调用不超过64条文本
    batch_size = 10
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(
            model=os.getenv("EMBEDDING_MODEL_ID"),
            input=batch
        )
        # 将响应转换为numpy数组列表并添加到embeddings列表中
        embeddings.extend([np.array(embedding.embedding) for embedding in response.data])

    return embeddings

# 假设text_chunks_dict是一个字典，键是块大小，值是文本块列表
chunk_embeddings_dict = {}
for size, chunks in tqdm(text_chunks_dict.items(), desc="Generating Embeddings"):
    chunk_embeddings_dict[size] = create_embeddings(chunks)

def cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度。

    参数：
    vec1 (np.ndarray): 第一个向量。
    vec2 (np.ndarray): 第二个向量。

    返回：
    float: 余弦相似度分数。
    """

    # 计算两个向量的点积
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# 获取相关的chunks
def retrieve_relevant_chunks(query, text_chunks, chunk_embeddings, k=5):
    """
    检索与查询最相关的前k个文本块

    参数：
    query (str): 用户查询
    text_chunks (List[str]): 文本块列表
    chunk_embeddings (List[np.ndarray]): 文本块的嵌入列表
    k (int): 返回的前k个块数量

    返回：
    List[str]: 最相关的文本块列表
    """
    # 为查询生成一个嵌入 - 将查询作为列表传递并获取第一个项目
    query_embedding = create_embeddings([query])[0]

    # 计算查询嵌入与每个块嵌入之间的余弦相似度
    similarities = [cosine_similarity(query_embedding, emb) for emb in chunk_embeddings]

    # 获取前k个最相似块的索引
    top_indices = np.argsort(similarities)[-k:][::-1]

    # 返回前k个最相关的文本块
    return [text_chunks[i] for i in top_indices]

# ============================从 JSON 文件加载验证数据=========================
with open('./data/val.json', encoding="utf-8") as f:
    data = json.load(f)

# 从验证数据中提取第一个查询
query = data[3]['question']

# 对于每个块大小，检索相关的文本块
retrieved_chunks_dict = {size: retrieve_relevant_chunks(query, text_chunks_dict[size], chunk_embeddings_dict[size]) for size in chunk_sizes}

# 打印块大小为 256 的检索到的文本块
print(f'Chunk Size 256 的检索到的文本块: {retrieved_chunks_dict[256]}')

# ===========================基于检索到的片段生成响应=========================
# AI 助手的系统提示
system_prompt = "你是一个AI助手，严格根据给定的上下文进行回答。如果无法直接从提供的上下文中得出答案，请回复：'我没有足够的信息来回答这个问题。'"

def generate_response(query, system_prompt, retrieved_chunks):
    """
    基于检索到的文本块生成 AI 回答。

    参数：
    query (str): 用户查询
    retrieved_chunks (List[str]): 检索到的文本块列表
    model (str): AI模型。

    返回：
    str: AI生成的回答。
    """
    # 将检索到的文本块合并为一个上下文字符串
    context = "\n".join([f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(retrieved_chunks)])

    # 通过组合上下文和查询创建用户提示
    user_prompt = f"{context}\n\nQuestion: {query}"

    # 使用指定模型生成AI回复
    response = client.chat.completions.create(
        model=os.getenv("LLM_MODEL_ID"),
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # 返回AI回复的内容
    return response.choices[0].message.content

# 为每个块大小生成 AI 回答
ai_responses_dict = {size: generate_response(query, system_prompt, retrieved_chunks_dict[size]) for size in chunk_sizes}

# 打印块大小为 256 的回答
print(f'Chunk Size 256 的回答: {ai_responses_dict[256]}')

# =========================评估响应质量=========================
# 定义评估评分系统的常量
SCORE_FULL = 1.0     # 完全匹配或完全令人满意
SCORE_PARTIAL = 0.5  # 部分匹配或部分令人满意
SCORE_NONE = 0.0     # 无匹配或不令人满意

# 定义严格的评估提示模板
FAITHFULNESS_PROMPT_TEMPLATE = """
评估 AI 回答与真实答案的一致性、忠实度。
用户查询: {question}
AI 回答: {response}
真实答案: {true_answer}

一致性衡量 AI 回答与真实答案中的事实对齐的程度，且不包含幻觉信息。
忠实度衡量的是AI的回答在没有幻觉的情况下与真实答案中的事实保持一致的程度。

指示：
- 严格使用以下值进行评分：
    * {full} = 完全一致，与真实答案无矛盾
    * {partial} = 部分一致，存在轻微矛盾
    * {none} = 不一致，存在重大矛盾或幻觉信息
- 仅返回数值评分（{full}, {partial}, 或 {none}），无需解释或其他附加文本。
"""

RELEVANCY_PROMPT_TEMPLATE = """
评估 AI 回答与用户查询的相关性。
用户查询: {question}
AI 回答: {response}

相关性衡量回答在多大程度上解决了用户的问题。

指示：
- 严格使用以下值进行评分：
    * {full} = 完全相关，直接解决查询
    * {partial} = 部分相关，解决了一些方面
    * {none} = 不相关，未能解决查询
- 仅返回数值评分（{full}, {partial}, 或 {none}），无需解释或其他附加文本。
"""

def evaluate_response(question, response, true_answer):
        """
        根据忠实度和相关性评估 AI 生成的回答质量

        参数：
        question (str): 用户的原始问题
        response (str): 被评估的 AI 生成的回答
        true_answer (str): 作为基准的真实答案

        返回：
        Tuple[float, float]: 包含 (忠实度评分, 相关性评分) 的元组。
                             每个评分可能是：1.0（完全匹配）、0.5（部分匹配）或 0.0（无匹配）。
        """
        # 格式化评估提示
        faithfulness_prompt = FAITHFULNESS_PROMPT_TEMPLATE.format(
                question=question,
                response=response,
                true_answer=true_answer,
                full=SCORE_FULL,
                partial=SCORE_PARTIAL,
                none=SCORE_NONE
        )

        relevancy_prompt = RELEVANCY_PROMPT_TEMPLATE.format(
                question=question,
                response=response,
                full=SCORE_FULL,
                partial=SCORE_PARTIAL,
                none=SCORE_NONE
        )

        # 模型进行忠实度评估
        faithfulness_response = client.chat.completions.create(
               model=os.getenv("LLM_MODEL_ID"),
                temperature=0,
                messages=[
                        {"role": "system", "content": "你是一个客观的评估者，仅返回数值评分。"},
                        {"role": "user", "content": faithfulness_prompt}
                ]
        )

        # 模型进行相关性评估
        relevancy_response = client.chat.completions.create(
                model=os.getenv("LLM_MODEL_ID"),
                temperature=0,
                messages=[
                        {"role": "system", "content": "你是一个客观的评估者，仅返回数值评分。"},
                        {"role": "user", "content": relevancy_prompt}
                ]
        )

        # 提取评分并处理潜在的解析错误
        try:
                faithfulness_score = float(faithfulness_response.choices[0].message.content.strip())
        except ValueError:
                print("Warning: 无法解析忠实度评分，将默认为 0")
                faithfulness_score = 0.0

        try:
                relevancy_score = float(relevancy_response.choices[0].message.content.strip())
        except ValueError:
                print("Warning: 无法解析相关性评分，将默认为 0")
                relevancy_score = 0.0

        return faithfulness_score, relevancy_score

# 第一条验证数据的真实答案
true_answer = data[3]['ideal_answer']

# 评估块大小为 256 和 128 的回答
faithfulness, relevancy = evaluate_response(query, ai_responses_dict[256], true_answer)
faithfulness2, relevancy2 = evaluate_response(query, ai_responses_dict[128], true_answer)

# 打印评估分数
print(f"忠实度评分 (Chunk Size 256): {faithfulness}")
print(f"相关性评分 (Chunk Size 256): {relevancy}")

print(f"\n")

print(f"忠实度评分 (Chunk Size 128): {faithfulness2}")
print(f"相关性评分 (Chunk Size 128): {relevancy2}")


