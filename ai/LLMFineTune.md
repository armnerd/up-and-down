# LLM 微调

## 1. huggingface

### 1.1. peft

> Parameter-Efficient Fine-Tuning

https://huggingface.co/docs/peft
https://github.com/huggingface/peft

### 1.2. LoRA

> Low-Rank Adaptation

https://huggingface.co/docs/peft/en/task_guides/lora_based_methods

### 1.3 Code

```python
// Datasets
from datasets import load_dataset
ds = load_dataset("food101")

// Train
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM, TrainingArguments, Trainer
peft_config = LoraConfig(
  task_type=TaskType.SEQ_2_SEQ_LM, 
  inference_mode=False,
  r=8, 
  lora_alpha=32,
  lora_dropout=0.1,
)
model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/mt0-large")
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
training_args = TrainingArguments(
    output_dir="your-name/bigscience/mt0-large-lora",
    learning_rate=1e-3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
trainer.train()

// Save
model.save_pretrained("output_dir")

// Inference
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
model = AutoPeftModelForCausalLM.from_pretrained("ybelkada/opt-350m-lora")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
```

## 2. LLaMA-Factory

https://github.com/hiyouga/LLaMA-Factory

* 北航开源的图形化微调工具

## 3. llama.cpp

https://github.com/ggerganov/llama.cpp

* 运行大模型做推理，还可以提供 api 服务
* 转换为 gguf 格式，然后可以量化为 4bit 版本

```bash
// build
make

// inference
./llama-cli -m your_model.gguf -p "You are a helpful assistant" -cnv

// server
./llama-server -m your_model.gguf --port 8080

// convert
python convert_hf_to_gguf.py

// quantize
./llama-quantize
```

## 4. tokenizer

https://tiktokenizer.vercel.app

* Tokens are the fundemantal unit, the “atom” of Large Language Models (LLMs). Tokenization is the process of translating strings (i.e. text) and converting them into sequences of tokens and vice versa.
* In the machine learning context, a token is typically not a word. It could be a smaller unit, like a character or a part of a word, or a larger one like a whole phrase. 
* Embeddings are dense, low-dimensional, continuous vector representations of tokens.
