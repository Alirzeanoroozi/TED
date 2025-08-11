from datasets import load_dataset, load_from_disk
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
import numpy as np
import wandb
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

DATA_PATH = "dlp/extended_json_data/train.jsonl"      # JSONL with {"sequence": "...", "spans": "11-12_34-34"}
def get_the_latest_checkpoint(output_dir="/home/aac/TED/new_Output"):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        raise ValueError(f"No checkpoint found in {output_dir}")
    
    steps = [int(d.split("-")[1]) for d in checkpoints]
    latest_index = np.argmax(steps)
    latest_checkpoint = checkpoints[latest_index]
    
    return os.path.join(output_dir, latest_checkpoint)

# model_name = get_the_latest_checkpoint()
model_name = "t5-3b" 

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 1) Number of parameters
num_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {num_params:,}")

# If you want the approximate RAM footprint when loaded as float32:
bytes_fp16 = num_params * 2
print(f"Approx memory (FP16): {bytes_fp16/1e9:.2f} GB")
model = model.to("cuda")

# exit()
CACHE_DIR = "extended_tokenized_ds/"
if os.path.isdir(CACHE_DIR):
    # 1) load pre-tokenized if it already exists
    tokenized = load_from_disk(CACHE_DIR)
else:
    # 2) otherwise build it and save it for next time
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
            max_length=1024,
            padding="max_length",
            truncation=True,
        )
        labels = tokenizer(
            examples["spans"],
            max_length=128,
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

    # 3) save to disk so you can reload instantly next time
    tokenized.save_to_disk(CACHE_DIR)

# now `tokenized` is ready to feed into your Trainer
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

def compute_metrics(eval_pred):
    preds = eval_pred.predictions
    labels = eval_pred.label_ids
    preds = np.clip(preds, 0, tokenizer.vocab_size - 1)

    # 4) Exact‐match on the decoded strings
    decoded_inputs  = tokenizer.batch_decode(tokenized["validation"]["input_ids"], skip_special_tokens=True)
    decoded_preds  = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Row-wise accuracy
    row_accuracies = []
    for i in range(len(decoded_inputs)):
        corr = 0
        for j in len(decoded_labels[i]):
            if j >= len(decoded_preds[i]):
                continue
            if decoded_preds[i][j] == decoded_labels[i][j]:
                corr += 1
        row_accuracies.append(float(corr / len(decoded_labels[i])))
    
    # 5) (Optional) log a small W&B table of raw examples
    table = wandb.Table(columns=["input", "prediction", "label", "token_accuracy"])
    for inp, pr, lab, acc in zip(decoded_inputs, decoded_preds, decoded_labels, row_accuracies):
        table.add_data(inp, pr, lab, acc)
    wandb.log({"eval_samples": table})
    
    return {"token_accuracy": sum(row_accuracies) / len(row_accuracies)}

training_args = Seq2SeqTrainingArguments(
    output_dir="continue/",
    run_name="Continue Training",
    
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1, # 4 GPUs * 2 Batch size * 2 accumulation_step = 12 Effective Batch Size

    # instead of evaluation_strategy="steps":
    eval_strategy="steps",    # run evaluation every `eval_steps`
    eval_steps=1,          
    # enables saving & evaluation at those steps
    save_steps=1,
    logging_steps=1,

    learning_rate=3e-5,
    weight_decay=0.01,
    predict_with_generate=True,
    fp16=True,
    num_train_epochs=3,
    save_total_limit=1,
    load_best_model_at_end=True,
)

# 9. Instantiate Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# kick off training
trainer.train()
