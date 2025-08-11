from datasets import load_dataset, load_from_disk
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import numpy as np
import wandb
import os
import torch
from typing import Optional
from safetensors.torch import load_file

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

DATA_PATH = "dlp/extended_json_data/train.jsonl"      # JSONL with {"sequence": "...", "spans": "11-12_34-34"}

def get_the_latest_checkpoint(output_dir="./qlora_checkpoints"):
    """Get the latest checkpoint from the checkpoints directory"""
    if not os.path.exists(output_dir):
        return None
    
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None
    
    steps = [int(d.split("-")[1]) for d in checkpoints]
    latest_index = np.argmax(steps)
    latest_checkpoint = checkpoints[latest_index]
    
    return os.path.join(output_dir, latest_checkpoint)

def setup_qlora_model(model_name: str, checkpoint_path: Optional[str] = None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 4-bit quantization configuration - More conservative
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,  # Disabled for stability
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32  # Use float32 for maximum stability
    )
    
    # Load model with quantization
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Very conservative LoRA configuration
    lora_config = LoraConfig(
        r=4,  # Very low rank
        lora_alpha=8,  # Very low alpha
        target_modules=["q", "v"],  # Only target key modules
        lora_dropout=0.0,  # No dropout
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        
        # Try loading from safetensors first (newer format)
        adapter_path_safetensors = os.path.join(checkpoint_path, "adapter_model.safetensors")
        
        if os.path.exists(adapter_path_safetensors):
            print(f"Loading from safetensors: {adapter_path_safetensors}")
            adapter_weights = load_file(adapter_path_safetensors)
            model.load_state_dict(adapter_weights, strict=False)

        else:
            print(f"Warning: No adapter weights found in {checkpoint_path}")
    
    # Print model info
    model.print_trainable_parameters()
    
    return model, tokenizer

# Check for latest checkpoint
checkpoint_path = get_the_latest_checkpoint()
model_name = "t5-3b"

# Setup model with QLoRA
model, tokenizer = setup_qlora_model(model_name, checkpoint_path)

CACHE_DIR = "extended_tokenized_ds/"
if os.path.isdir(CACHE_DIR):
    tokenized = load_from_disk(CACHE_DIR)
else:
    raw_ds = load_dataset(
        "json",
        data_files={
            "train": DATA_PATH,
            "validation": DATA_PATH.replace("train", "validation"),
        },
    )

    def preprocess(examples):
        model_inputs = tokenizer(
            examples["sequence"],
            max_length=2048,
            padding="max_length",
            truncation=True,
        )
        labels = tokenizer(
            examples["spans"],
            max_length=256,
            padding="max_length",
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = raw_ds.map(
        preprocess,
        batched=True,
        remove_columns=raw_ds["train"].column_names,
    )

    tokenized.save_to_disk(CACHE_DIR)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

def compute_metrics(eval_pred):
    preds = eval_pred.predictions
    labels = eval_pred.label_ids
    preds = np.clip(preds, 0, tokenizer.vocab_size - 1)

    decoded_inputs  = tokenizer.batch_decode(tokenized["validation"]["input_ids"], skip_special_tokens=True)
    decoded_preds  = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Row-wise accuracy
    row_accuracies = []
    for i in range(len(decoded_inputs)):
        corr = 0
        for j in range(len(decoded_labels[i])):
            if j >= len(decoded_preds[i]):
                continue
            if decoded_preds[i][j] == decoded_labels[i][j]:
                corr += 1
        if len(decoded_labels[i]) > 0:
            row_accuracies.append(float(corr / len(decoded_labels[i])))
    
    table = wandb.Table(columns=["input", "prediction", "label", "token_accuracy"])
    for inp, pr, lab, acc in zip(decoded_inputs, decoded_preds, decoded_labels, row_accuracies):
        table.add_data(inp, pr, lab, acc)
    wandb.log({"eval_samples": table})
    
    return {"token_accuracy": sum(row_accuracies) / len(row_accuracies)}

training_args = Seq2SeqTrainingArguments(
    output_dir="qlora_checkpoints/",
    run_name="QLoRA Training",
    
    per_device_train_batch_size=4,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=8,
    
    eval_strategy="steps",
    eval_steps=50,
    save_steps=50,
    logging_steps=5,

    learning_rate=5e-5,
    weight_decay=0.0,
    predict_with_generate=True,
    
    num_train_epochs=1,
    save_total_limit=1,
    
    max_grad_norm=0.1,
    
    resume_from_checkpoint=checkpoint_path if checkpoint_path else None,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
