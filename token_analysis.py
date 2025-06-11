# This script is used to analyze the token length of the labels we want the model to output
# It is used to determine the maximum length of output tokens for model to generate
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

labels = ["in-favor", "against", "neutral-or-unclear"]
for label in labels:
    tokens = tokenizer.tokenize(label)
    token_count = len(tokenizer(label)["input_ids"])
    print(f"'{label}' -> {tokens} -> {token_count} tokens")