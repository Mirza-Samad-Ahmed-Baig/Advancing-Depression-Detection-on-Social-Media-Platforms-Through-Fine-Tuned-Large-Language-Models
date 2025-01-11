%%capture
%pip install accelerate peft bitsandbytes transformers trl
!huggingface-cli login
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

from google.colab import drive
drive.mount('/content/drive')

# Model from Hugging Face hub
base_model = "meta-llama/Llama-2-7b-chat-hf"

# New instruction dataset
dataset_name = "yourpath/dataset_name"

# Fine-tuned model
new_model = "/yourpath/foldername/pad_model"

training_dataset = load_dataset(dataset_name, split="train")
evaluation_dataset = load_dataset(dataset_name, split="validation")

compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map={"": 0}
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.unk_token
model.config.pad_token_id = tokenizer.pad_token_id # updating model config
tokenizer.padding_side = 'right' # padding to right (otherwise SFTTrainer shows warning)

peft_params = LoraConfig(
    lora_alpha=128,
    lora_dropout=0,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

training_params = TrainingArguments(
    output_dir="/yourpath/foldername/output_dir",
    num_train_epochs=20, 
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    eval_strategy="epoch",
    logging_strategy = "epoch",
    do_eval=True,
    optim="paged_adamw_32bit",
    save_strategy="epoch",
    learning_rate=1e-4,
    weight_decay=0,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=training_dataset,
    eval_dataset=evaluation_dataset,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)
trainer.train()

trainer.model.save_pretrained("/yourpath/foldername/pad_model")
trainer.tokenizer.save_pretrained("/yourpath/foldername/pad_model")

from tensorboard import notebook
log_dir = "/yourpath/foldername/output_dir/runs"
notebook.start("--logdir {} --port 4000".format(log_dir))

# Empty VRAM
del model
del pipe
del trainer
import gc
gc.collect()
gc.collect()

# Reload model in FP16 and merge it with LoRA weights
#merging and pushing

base_model = "meta-llama/Llama-2-7b-chat-hf"

base_modell = AutoModelForCausalLM.from_pretrained(
    base_model,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map={"": 0},
)

saved_model = "/yourpath/foldername/pad_model"

model = PeftModel.from_pretrained(base_modell, saved_model)
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.unk_token
model.config.pad_token_id = tokenizer.pad_token_id # updating model config
tokenizer.padding_side = 'right' # padding to right (otherwise SFTTrainer shows warning)


logging.set_verbosity(logging.CRITICAL)

prompt = "Well well well, see who slept again for 14 hours :( ?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=2000)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])
