# %%
# !pip install datasets transformers bitsandbytes peft evaluate

# %%
import torch
import copy
from datasets import load_dataset

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer
)

from dataclasses import dataclass
import os

# %%
@dataclass
class DataClass:
    MODEL_PATH = "Qwen/Qwen2-0.5B-Instruct"
    MAX_LENGTH = 96
    EPOCH = 3
    LORA_RANK = 16
    LORA_ALPHA = 4 * LORA_RANK          # 2 * LORA_RANK
    LORA_DROPOUT = 0.5
    LORA_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                    "lm_head"]
    LR = 5e-5
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps'
    USE_LORA = True
    MODEL_SAVE_FOLDER = os.path.join('smol_weights', 'LORA' if USE_LORA else 'RegularFinetune')
    

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
    attn_implementation='eager', #NOTE: MPS does ont support 'flash_attention_2',
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    # quantization_config=quant_config
)

# %%
if DataClass.USE_LORA:
    print("Using LORA instead of full-finetuning")
    # Freezing model weights
    model.requires_grad = False
    
    lora_config = LoraConfig(
        r = DataClass.LORA_RANK,
        lora_alpha = DataClass.LORA_ALPHA,
        lora_dropout = DataClass.LORA_DROPOUT,
        bias = "none",
        target_modules = DataClass.LORA_MODULES,
        task_type = TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

# %%
print("Model structure")
print(model)

# %%
def inference(input_text):
    input_ids = tokenizer(input_text, return_tensors="pt")
    # print(input_ids.keys())
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
    return tokenizer.decode(outputs[0])

# %%
# input_text = "Write a poem in machine learning.\n"
# print(inference(input_text=input_text))

# %%
# Let's see the chat template

prompt = "Write a poem in machine learning."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print("Chat Template:")
print(text)
print('-----')
print("Output:")
print(inference(text))
print()

# %%
train_dataset = load_dataset("json", data_files="dataset/train.json")
test_dataset = load_dataset("json", data_files="dataset/valid.json")

# %%
print(train_dataset)
print(test_dataset)

# %%
def prompt_formatter(data):
    """Formatting the prompt"""
    data = \
f'''<|im_start|>system
You are an advanced language model adept at interpreting and refining noisy or imperfect user inputs.
Given user data, your task is to accurately extract the intended question and provide precise answers or predictions, even if the input contains errors or discontinuities.<|im_end|>
<|im_start|>user
{data['input_disfluent']}<|im_end|>
<|im_start|>assistant
{data['output_original']}<|im_end|>'''

    return {'sentence': data, 'input_ids': '', 'attention_mask': '', 'labels': ''}

print(prompt_formatter(train_dataset['train'][0])['sentence'])

# %%
def batch_tokenizer(batch):
    """Tokenization of data"""
    model_inputs = tokenizer(
        batch["sentence"],
        max_length=DataClass.MAX_LENGTH,
        truncation=True,
        padding='max_length'
    )
    # HF automatically performs right shift
    model_inputs['labels'] = copy.deepcopy(model_inputs['input_ids'])
    return model_inputs

# %%
train_dataset = train_dataset.map(prompt_formatter)
train_dataset = train_dataset.map(batch_tokenizer, batched=True, remove_columns = [])
test_dataset = test_dataset.map(prompt_formatter)
test_dataset = test_dataset.map(batch_tokenizer, batched=True, remove_columns = [])

# %%
data_collator = DataCollatorForSeq2Seq(
    model = model,
    tokenizer = tokenizer,
    max_length = DataClass.MAX_LENGTH,
    pad_to_multiple_of = 2,
    padding = 'max_length'
)

training_args = TrainingArguments(
    disable_tqdm=False,
    output_dir = DataClass.MODEL_SAVE_FOLDER,
    overwrite_output_dir=True,
    # fp16=True,
    save_only_model=True,               # Only saving model, unresumable
    per_device_eval_batch_size=1,
    learning_rate=DataClass.LR,
    num_train_epochs=DataClass.EPOCH,
    logging_strategy='steps',
    logging_steps=128,
    eval_strategy='steps',
    eval_steps=7182//8,
    save_strategy='steps',
    save_steps=7182//8,
    push_to_hub=False,
    weight_decay=0.9,           # Overfit?
    report_to=[]
)

# %%
# Check some examples

for i in range(3):
    print(f"Example {i+1}:")
    print(tokenizer.decode(train_dataset['train'][i]['input_ids']))
    print()

# %%
trainer = Trainer(
    model = model,
    tokenizer= tokenizer,
    args = training_args,
    train_dataset= train_dataset['train'],
    eval_dataset=test_dataset['train'],
    data_collator=data_collator
)

model.config.use_cache = False
trainer.train()

# %%



