from transformers import AutoTokenizer,AutoModelForCausalLM
import os
import torch

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_path = os.path.expanduser('~/models/Qwen/Qwen3-0.6B')

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=device,
    torch_dtype=torch.bfloat16,  # 使用BF16
    # attn_implementation="flash_attention_2",  # 如果支持Flash Attention
    )
tokenizer = AutoTokenizer.from_pretrained(model_path)

import os
from datasets import load_dataset
data_dir = os.path.expanduser("~/datasets/alpaca-gpt4-data-zh")
ds = load_dataset("json",data_dir=data_dir,split="train")


def process_func(example):
    f"""
    处理数据集，用来把数据改造成适合训练的格式.
    对话数据集，这里的每个example都是一次对话，只有问题&回答
    这里需要加上Human和Assistant标签
    Args:
        example字典结构:
            'instruction':字符串
            'input':str
            'output':str
        使用tokenizer的apply_chat_template方法
    """
    MAX_LENGTH=256
    user_content = example['instruction']
    if example['input'].strip():
        user_content += "\n" + example['input']
    
    messages = [
        {"role":"user","content":user_content},
        {"role":"assistant","content":example['output']}
    ]
    # 使用chat template
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False, # 返回文本而不是token ids
        add_generation_prompt=False
    )
    
    # 分别处理用户和助手部分
    user_messages  = [{"role":"user","content":user_content}]
    user_text = tokenizer.apply_chat_template(
        user_messages,
        tokenize=False, # 返回文本而不是token ids
        add_generation_prompt=True # 添加assistant开始标记
    )
    # tokenize
    full_tokens = tokenizer(full_text,add_special_tokens=False)
    user_tokens = tokenizer(user_text,add_special_tokens=False)
    
    input_ids = full_tokens['input_ids']
    attention_mask = full_tokens['attention_mask']
    # 创建labels 用来计算loss
    labels = [-100] * len(user_tokens['input_ids']) + \
        input_ids[len(user_tokens['input_ids']):]
    
    # 长度截断
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    
    return {
        "input_ids":input_ids,
        "attention_mask":attention_mask,
        "labels":labels
    }
    

# 转化数据,删除原始列名，默认是保留的(instruction,input,output)
train_ds = ds.map(process_func,remove_columns = ds.column_names)


from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="model/qwen3-0.6B",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=2,
    weight_decay=1e-4,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=20,
    logging_strategy='steps',
    bf16=True,  # 启用BF16
    fp16=False,  # 确保fp16关闭
)


from transformers import Trainer
from transformers import DataCollatorForSeq2Seq
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    processing_class=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer,padding=True),
)

trainer.train()