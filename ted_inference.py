from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import numpy as np
import os
import json
import pandas as pd
import torch.nn.functional as F

DATA_PATH = "dlp/extended_json_data/validation.jsonl"      # JSONL with {"sequence": "...", "spans": "11-12_34-34"}
data = {
    "sequence": [],
    "spans": [],
}
with open(DATA_PATH, "r") as f:
    for line in f:
        line_dict = json.loads(line.strip())
        data["sequence"].append(line_dict["sequence"])
        data["spans"].append(line_dict["spans"])

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_the_latest_checkpoint(output_dir="/home/aac/TED/new_Output"):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        raise ValueError(f"No checkpoint found in {output_dir}")
    
    steps = [int(d.split("-")[1]) for d in checkpoints]
    latest_index = np.argmax(steps)
    latest_checkpoint = checkpoints[latest_index]
    
    return os.path.join(output_dir, latest_checkpoint)

model_name = get_the_latest_checkpoint()
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

# 1) Number of parameters
num_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {num_params:,}")

# If you want the approximate RAM footprint when loaded as float32:
bytes_fp32 = num_params * 4
print(f"Approx memory (FP32): {bytes_fp32/1e9:.2f} GB")

def preprocess(examples):
    model_inputs = tokenizer(
        examples["sequence"],
        max_length=1024,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    labels = tokenizer(
        examples["spans"],
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    model_inputs["labels"] = labels["input_ids"]
    # Move all tensors to device
    for k in model_inputs:
        if isinstance(model_inputs[k], torch.Tensor):
            model_inputs[k] = model_inputs[k].to(device)
    return model_inputs

def compute_metrics(input_ids, preds, labels):
    decoded_inputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    decoded_preds  = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_inputs_raw = tokenizer.batch_decode(input_ids)
    decoded_preds_raw  = tokenizer.batch_decode(preds)
    decoded_labels_raw = tokenizer.batch_decode(labels)
    
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
    
    pd.DataFrame({
        "input_raw": decoded_inputs_raw,
        "prediction_raw": decoded_preds_raw,
        "label_raw": decoded_labels_raw,
        "input": decoded_inputs,
        "prediction": decoded_preds,
        "label": decoded_labels,
        "token_accuracy": row_accuracies
    }).to_csv("results.csv", index=False)

model_inputs = preprocess(data)

batch_size = 32  # or any batch size you prefer
input_ids = model_inputs["input_ids"]
attention_mask = model_inputs["attention_mask"]

all_preds = []
for i in range(0, input_ids.shape[0], batch_size):
    batch = {
        "input_ids": input_ids[i:i+batch_size],
        "attention_mask": attention_mask[i:i+batch_size]
    }

    batch_preds = model.generate(**batch, max_length=128)
    if batch_preds.shape[1] < 128:
        batch_preds = F.pad(batch_preds, (0, 128 - batch_preds.shape[1]), value=0)
    elif batch_preds.shape[1] > 128:
        batch_preds = batch_preds[:, :128]
        
    all_preds.append(batch_preds)
    print(i, "from", input_ids.shape[0], "done")
preds = torch.cat(all_preds, dim=0).cpu().numpy()
print(preds.shape)

compute_metrics(model_inputs["input_ids"].cpu().numpy(), preds, model_inputs["labels"].cpu().numpy())