# %%
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    # BitsAndBytesConfig,
)

import json
import evaluate

# %%
from dataclasses import dataclass
import os

@dataclass
class DataClass:
    MODEL_PATH = ["weights/RegularFinetune/checkpoint-897", "Qwen/Qwen2-0.5B-Instruct"][0]
    MAX_LENGTH = 96
    EPOCH = 3
    LORA_RANK = 2
    LORA_ALPHA = 2 * LORA_RANK
    LORA_DROPOUT = 0.5
    LORA_MODULES = ["o_proj", "qjv_proj", "gate_up_proj"]
    LR = 5e-5
    MODEL_SAVE_FOLDER = '/content/drive/MyDrive/weights'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps'

# Macbook MPS
if DataClass.DEVICE == 'mps':
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    # Optional: Turnning off tokenizer parallelism to avoid stuck
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"

# %%
model_config = AutoConfig.from_pretrained(
    DataClass.MODEL_PATH,
    trust_remote_code = True,
    attn_implementation = 'eager', #'flash_attention_2'
)

tokenizer = AutoTokenizer.from_pretrained(
    DataClass.MODEL_PATH,
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
    DataClass.MODEL_PATH,
    device_map=DataClass.DEVICE,
    low_cpu_mem_usage=True,
    # load_in_8bit=True,
    # load_in_4bit=True,
    attn_implementation='eager', #'flash_attention_2',
    torch_dtype=torch.bfloat16, # NOTE: MPS does not support torch.bfloat16 finetuning
    trust_remote_code=True,
    # quantization_config=quant_config
)

# %%
def inference(input_text):
    input_ids = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(
        # **input_ids
        max_new_tokens=100,
        do_sample=False,
        num_beams=1,
        temperature=None,
        top_k=None,
        top_p=None,
        input_ids=input_ids['input_ids'].to(DataClass.DEVICE),
        attention_mask=input_ids['attention_mask'].to(DataClass.DEVICE)
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
with open("dataset/valid.json", "r") as f:
    valid_data = json.load(f)

# %%
for i in range(10):
    q = valid_data[i]['input_disfluent']
    a = valid_data[i]['output_original']

    print("LLM:", prompter(q))
    print("ANS:", a)
    print('---')

# %%
bleu_metric = evaluate.load("bleu")
tot_bleu = 0.
model_io = []

for idx, data in enumerate(valid_data):
    ques = data['input_disfluent']
    ref = data['output_original']
    pred = prompter(ques)
    bleu = bleu_metric.compute(predictions=[pred], references=[ref])['bleu']
    tot_bleu += bleu
    model_io.append({"input_ques": ques, "output_pred": pred, "output_ground": ref, "bleu": bleu})
    print(f"\rData {idx+1}, BLEU: {tot_bleu/(idx+1)}", end="")

model_io.append({"AVG_BLEU": tot_bleu/len(valid_data)})
with open("outputs/valid_io.json", "w") as f:
    json.dump(model_io, f, indent=2)

print()
print(tot_bleu/len(valid_data))

# %%
with open("dataset/train.json", "r") as f:
    train_data = json.load(f)

# %%
bleu_metric = evaluate.load("bleu")
tot_bleu = 0.
model_io = []

for idx, data in enumerate(train_data):
    ques = data['input_disfluent']
    ref = data['output_original']
    pred = prompter(ques)
    bleu = bleu_metric.compute(predictions=[pred], references=[ref])['bleu']
    tot_bleu += bleu
    model_io.append({"input_ques": ques, "output_pred": pred, "output_ground": ref, "bleu": bleu})
    print(f"\rData {idx+1}, BLEU: {tot_bleu/(idx+1)}", end="")

model_io.append({"AVG_BLEU": tot_bleu/len(valid_data)})
with open("outputs/train_io.json", "w") as f:
    json.dump(model_io, f, indent=2)

print()
print(tot_bleu/len(train_data))

# %%



