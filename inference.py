# %%
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    # BitsAndBytesConfig,
)

import json
import os
import evaluate

# %%
def get_device():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    # Macbook MPS
    elif torch.backends.mps.is_available():
        device = 'mps'
    return device


# Macbook MPS
if get_device() == 'mps':
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    # Optional: Turnning off tokenizer parallelism to avoid stuck
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"


# %%
model_config = AutoConfig.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    trust_remote_code = True,
    attn_implementation = 'eager', #'flash_attention_2'
)

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    trust_remote_code = True
)

tokenizer.pad_token = tokenizer.eos_token

# quant_config = BitsAndBytesConfig(
#     load_in_4bit = True,
#     bnb_4bit_quant_type="n4f",
#     bnb4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=True
# )

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    device_map=get_device(),
    low_cpu_mem_usage=True,
    # load_in_8bit=True,
    # load_in_4bit=True,
    attn_implementation='eager', #'flash_attention_2',
    torch_dtype=torch.bfloat16, # NOTE: MPS does not support torch.bfloat16 finetuning
    trust_remote_code=True,
    # quantization_config=quant_config
)

model.load_adapter('weights/LORA/checkpoint-1794')

# %%
def inference(input_text):
    input_ids = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(
        # **input_ids
        max_new_tokens=100,
        # do_sample=False,
        # num_beams=1,
        # temperature=None,
        # top_k=None,
        # top_p=None,
        input_ids=input_ids['input_ids'].to(get_device()),
        attention_mask=input_ids['attention_mask'].to(get_device())
    )
    # Only generate output
    input_token_len = input_ids['input_ids'].shape[-1]
    return tokenizer.decode(outputs[0][input_token_len:], skip_special_tokens=True)

# %%
def prompter(question):
    prompt = f'''<|im_start|>system
You are an advanced language model adept at interpreting and refining noisy or imperfect user inputs.
Given user data, your task is to accurately extract the intended question and provide precise answers or predictions, even if the input contains errors or discontinuities.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
'''
    # print(prompt)
    return inference(prompt)

# %%
# The question has to be given here

prompter("What Microsoft err I mean Goldman Sachs CEO is also an alumni of the University of Chicago?")

# %%
with open("inference.json", "r") as f:
    inp_data = json.load(f)

# print(inp_data)

# %%
outputs = []
for data in inp_data:
    outputs.append(prompter(data))

# %%
# print(outputs)

# %%
with open("outputs/inference_output.json", "w") as f:
    json.dump(outputs, f, indent=2)

# %%



