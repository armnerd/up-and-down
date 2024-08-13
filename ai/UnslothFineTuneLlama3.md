# 使用 unsloth 微调 Llama 3

> https://huggingface.co/blog/mlabonne/sft-llama3

## 微调

* 大学生岗前培训，从通用技能到专项技能
* 大模型通常有非常多的节点，如果每个都调整，费时费力还可能影响了通用的功能
* 模型如果足够大，摆在面前的第一个问题就是要有足够的内存存放，which is extremely expensive
* 那我们就只调节一部分节点，which called LoRA
* 如果想继续节省调试需要的内存，就要用时间换空间，which called QLoRA

## terms

* LoRA === **Low-Rank Adaptation** [ 16 bit ]
* QLoRA === **Quantization-aware Low-Rank Adaptation** [ 4 bit ]
* GGML === **GPT-Generated Model Language**
* GGUF === **GPT-Generated Unified Format**

## 上链接

```bash
// unsloth
https://unsloth.ai/

// unsloth github
https://github.com/unslothai/unsloth
```

## 原材料

* 开源大模型 Llama 3
* 数据集 dataset of instructions and answers

## SFT 工艺

> Supervised Fine-Tuning

### 把大象装冰箱拢共需要多少步？

1. 加载预训练模型
2. 加载数据集
3. 初始化 trainer 并开始训练
4. 测试训练成果
5. 保存模型到本地

## 上代码

```python
# 安装 package
pip install unsloth

# import
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
import torch

# 加载模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# 加载数据集
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass
dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

# 初始化 trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field='text',
    max_seq_length=max_seq_length,
    packing=False,
    args = TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps=logging_steps,
        optim='adamw_8bit',
        weight_decay=0.01,
        lr_scheduler_type='linear',
        seed=seed,
        output_dir='outputs',
        save_steps=save_steps,
        max_steps=max_steps
    )
)

# 训练
trainer_stats = trainer.train()

# 验证
FastLanguageModel.for_inference(model)
inputs = tokenizer(
[
    alpaca_prompt.format(
        "Continue the fibonnaci sequence.", # instruction
        "1, 1, 2, 3, 5, 8", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
tokenizer.batch_decode(outputs)

# 保存
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
```