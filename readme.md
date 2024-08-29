# Chata.AI Question Rewrite

The task of the project is to rewrite a question in a way that removes the irrelevant wordings, fixes typos, while keeping the original context of the question.


## Model Information
A small language model (SLM) is used to solve this task. `Qwen-2:0.5b` has been used to solve this task. Being a SLM, the Qwen-2 can be used to perform 'regular finetune' instead of paremeter efficient finetuning (example: LORA, QLORA). My experiments show that 'regular finetune' is exceptionally better than LORA.

## Memory Requirement

Being SLM, Qwen-2 0.5B needs `~1 GB` of memory in `16-bit` float. The regular finetuning took `~ 12 GB` of memory due to optimizer and other overheads.

For industrial inference `Q8 (8-bit quantization)` could be one of the best option (as the model is already small) to save memory and increase inference speed.

## Finetuning

The finetuning was done with 3 iterations. Below is the loss of train and validate dataset:

Epoch | Train Loss | Valid Loss | LR
--- | --- | --- | ---
1 | 0.672 | 0.642 | 3.33<sup>-5</sup>
2 | 0.601 | 0.625 | 1.67<sup>-5</sup>
3 | 0.548 | 0.626 | 1.11<sup>-7</sup>

Epoch 3 shows the sign of overfitting in training dataset (as training loss is less than validation loss). Therefore, the weights at the end of Epoch 2 was used for the final experiment. 

## Evaluation

The model was evaluated with BLEU metric. The result is given below:

Train Data | Validation Data
--- | ---
0.889 | 0.861

Based on the evaluation, it can be validated that the trained model is not overconfident on the training data, as it contains a balance of performance in the unseen validation data. Note that more evaluation techniques could be used. Also, more ground examples could change the norm of the result.

## Steps for Improvement

As this work used an 0.5B SLM, using a higher parameter model would increase performance (Example: `Phi-3.5:3.8b`, `LLama-3.1:8b`). I did further experiment with smaller models (`smollm:135m`) and verified that smaller models shows less linguistic understanding. This is already evaluated by the LLM scaling laws.

## Finetuned Weights

Finetuned weights can be downloaded from the google drive url:
[Google Drive](https://drive.google.com/file/d/13_P4NyCM4Zw-JcU-aGXxBWuA7d1JhIu2/view?usp=sharing).

The weights should be inside the project directory (as shown in directory structure).

## Directory Structure

The directory structure with explanation is given as follows:

```
├── datamod.ipynb       # Used to convert downloaded dataset into huggingface requirement
├── dataset             # Datasets
│   ├── train.json
│   └── valid.json
├── eval.ipynb          # Jupyter notebook to run the metrics on the trained LLM output
├── eval.py             # Python code to run the metrics on the trained LLM output (converted)
├── finetune.ipynb      # Jupyter notebook to train LLM
├── finetune.py         # Python code to run the train LLM (converted)
├── inference.ipynb     # Jupyter notebook to get result form trained LLM
├── inference.py        # Jupyter notebook to get result form trained LLM (converted)
├── readme.md
├── outputs
│   ├── train_io.json   # Produced BLEU score on train data (generated from eval.py/ipynb)
│   ├── valid_io.json   # Produced BLEU score on test data (generated from eval.py/ipynb)
│   └── valid_io.json   # Inference output (generated from inference.py/ipynb)
├── inference.json      # Inference input questions as json list for testing
├── task-description.pdf
└── weights
    └── LORA
        ├── checkpoint-1794     # The saved weights from finetuning
        │   ├── README.md
        │   ├── adapter_config.json
        │   ├── adapter_model.safetensors
        │   ├── added_tokens.json
        │   ├── merges.txt
        │   ├── optimizer.pt
        │   ├── rng_state.pth
        │   ├── scheduler.pt
        │   ├── special_tokens_map.json
        │   ├── tokenizer.json
        │   ├── tokenizer_config.json
        │   ├── trainer_state.json
        │   ├── training_args.bin
        │   └── vocab.json
        |
```

## Model Inference

`inference.ipynb/.py` can be used to test the finetuned model. The input questions should be pasted in `inference.json` file as list. The produced outputs are saved in `outputs/inference_output.json` file.