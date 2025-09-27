import os



relative_path = '~/datasets/Anthropic/hh-rlhf'
abs_path = os.path.expanduser(relative_path)

# huggingface dataset
from datasets import load_dataset

dataset = load_dataset(abs_path, split='train')

print(dataset[0])