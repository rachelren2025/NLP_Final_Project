import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "auto"
model_path = "granite-3.0-8b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
model.eval()

chat = [
    { "instruction": "You will receive a message representing the context or facts of a legal case, followed by five possible legal holdings. Your task is to select the holding that is most relevant and aligns with the legal principles or facts in the message. Output only one number and nothing else at all.", "content": "Please list one IBM Research laboratory located in the United States. You should only output its name and location." 
     
     },
]
chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
# tokenize the text
input_tokens = tokenizer(chat, return_tensors="pt").to(device)
# generate output tokens
output = model.generate(**input_tokens, 
                        max_new_tokens=100)
# decode output tokens into text
output = tokenizer.batch_decode(output)
# print output
print(output)