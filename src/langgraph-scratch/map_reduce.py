from collections import Counter
import re
from functools import reduce

texts = [
    "I love AI and AI loves me.",
    "AI is the future of humanity.",
    "Humans and AI should work together."
]

# Map：每段统计词频
def map_step(text):
    words = re.findall(r'\w+', text.lower())
    return Counter(words)

# 分布到多个机器/线程来运行
mapped = [map_step(t) for t in texts]

# Reduce：合并所有段落的词频
def reduce_step(mapped_results):
    return reduce(lambda x, y: x + y, mapped_results)

final_result = reduce_step(mapped)

print(final_result)


import os

key = os.getenv("TAVILY_API_KEY")
print(f'key :{key}')# langchain