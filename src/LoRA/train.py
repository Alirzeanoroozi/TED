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
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="../configs/config_1.yaml")
args = parser.parse_args()

config = yaml.safe_load(open(args.config, "r"))

os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]

def setup_qlora_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 4-bit quantization configuration - More conservative
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config["load_in_4bit"],
        bnb_4bit_use_double_quant=config["bnb_4bit_use_double_quant"],  # Disabled for stability
        bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=config["bnb_4bit_compute_dtype"]  # Use float32 for maximum stability
    )
    
    # Load model with quantization
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # More effective LoRA configuration
    lora_config = LoraConfig(
        r=config["r"],                    # Higher rank for better expressiveness
        lora_alpha=config["lora_alpha"],           # Higher alpha (typically 2*r)
        target_modules=config["target_modules"],  # Target more attention modules
        lora_dropout=config["lora_dropout"],        # Small dropout for regularization
        bias=config["bias"],
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print model info
    model.print_trainable_parameters()
    
    return model, tokenizer

# Setup model with QLoRA
model, tokenizer = setup_qlora_model(config["model_name"])

CACHE_DIR = "extended_tokenized_ds/"
if os.path.isdir(CACHE_DIR):
    tokenized = load_from_disk(CACHE_DIR)
else:
    raw_ds = load_dataset(
        "json",
        data_files={
            "train": config["train_path"],
            "validation": config["validation_path"],
        },
    )

    def preprocess(examples):
        model_inputs = tokenizer(
            examples["Sequence"],
            max_length=2048,
            padding="max_length",
            truncation=True,
        )
        labels = tokenizer(
            examples["label"],
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
    output_dir=config["output_dir"],
    run_name=config["run_name"],
    
    per_device_train_batch_size=config["per_device_train_batch_size"],
    per_device_eval_batch_size=config["per_device_eval_batch_size"],
    gradient_accumulation_steps=config["gradient_accumulation_steps"],
    
    eval_strategy="steps",
    eval_steps=config["eval_steps"],
    save_steps=config["save_steps"],
    logging_steps=config["logging_steps"],

    learning_rate=float(config["learning_rate"]),
    weight_decay=float(config["weight_decay"]),
    predict_with_generate=True,
    
    num_train_epochs=config["num_train_epochs"],
    save_total_limit=config["save_total_limit"],
    
    max_grad_norm=config["max_grad_norm"],
    dataloader_pin_memory=config["dataloader_pin_memory"],
    dataloader_num_workers=config["dataloader_num_workers"],
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
