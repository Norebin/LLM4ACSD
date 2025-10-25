import pandas as pd
from datasets import Dataset
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig

def process_func(example):
    MAX_LENGTH = 4096
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer((f"<｜begin▁of▁sentence｜>User: {example['instruction']+example['input']}\n"
                             f"Assistant: "
                            ).strip(),
                            add_special_tokens=False)
    response = tokenizer(f"{example['output']}<｜end▁of▁sentence｜>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

model_path = '/model/LiangXJ/Model/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
tokenizer.padding_side = 'right'
df = pd.read_json('/model/lxj/LLaMA-Factory/data/codesmell/SFT_code_smell_alpaca.json')
ds = Dataset.from_pandas(df)
tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
model.enable_input_require_grads()
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj", 'gate_proj', 'up_proj', 'down_proj'],  # 现存问题只微调部分演示即可
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=16, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
args = TrainingArguments(
    output_dir="/model/LiangXJ/Model/PEFT/deepseekcoder-V2-16b-instruct/lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=1000,
    learning_rate=1e-5,
    save_on_each_node=True,
    gradient_checkpointing=True,
    use_cache=False
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
trainer.train()