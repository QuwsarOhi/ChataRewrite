# Chata.AI Question Rewrite

The task of the project is to rewrite a question in a way that removes the irrelevant wordings, fixes typos, while keeping the original context of the question.


## Model Information
A small language model (SLM) is used to solve this task. `Qwen-2 0.5B` has been used to solve this task. Being a SLM, the Qwen-2 can be used to perform 'regular finetune' instead of paremeter efficient finetuning (example: LORA, QLORA). My experiments show that 'regular finetune' is exceptionally better than LORA.

## Memory Requirement

Being SLM, Qwen-2 0.5B needs `~1 GB` of memory in `16-bit` float. The regular finetuning took `~ 12 GB` of memory due to optimizer and other overheads.

## Directory Structure

The directory structure with explanation is given as follows:

```
.
├── datamod.ipynb       # Used to convert the dataset JSON file into format that can be used by Huggingface Datasets
├── dataset
│   ├── train.json
│   └── valid.json
├── evaluate.ipynb
├── finetune.ipynb
├── finetune.py
├── outputs
│   ├── train_io.json
│   └── valid_io.json
├── readme.md
└── weights
    ├── LORA
    │   └── checkpoint-897
    │       ├── README.md
    │       ├── adapter_config.json
    │       ├── adapter_model.safetensors
    │       ├── added_tokens.json
    │       ├── merges.txt
    │       ├── optimizer.pt
    │       ├── rng_state.pth
    │       ├── scheduler.pt
    │       ├── special_tokens_map.json
    │       ├── tokenizer.json
    │       ├── tokenizer_config.json
    │       ├── trainer_state.json
    │       ├── training_args.bin
    │       └── vocab.json
    └── RegularFinetune
        └── checkpoint-897
            ├── added_tokens.json
            ├── config.json
            ├── generation_config.json
            ├── merges.txt
            ├── model.safetensors
            ├── optimizer.pt
            ├── rng_state.pth
            ├── scheduler.pt
            ├── special_tokens_map.json
            ├── tokenizer.json
            ├── tokenizer_config.json
            ├── trainer_state.json
            ├── training_args.bin
            └── vocab.json
```