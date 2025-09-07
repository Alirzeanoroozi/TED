from datasets import load_dataset, load_from_disk
import numpy as np
import wandb
import os
import yaml
import argparse
import torch
import torch.nn as nn

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/transformer_config.yaml")
args = parser.parse_args()

config = yaml.safe_load(open(args.config, "r"))

os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]

class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers,
        num_decoder_layers,
        emb_size,
        nhead,
        vocab_size,
        dim_feedforward=2048,
        dropout=0.1,
        pad_token_id=0,
        src_max_seq_len=2048,
        tgt_max_seq_len=128,
    ):
        super(Seq2SeqTransformer, self).__init__()
        self.src_tok_emb = nn.Embedding(vocab_size, emb_size, padding_idx=pad_token_id)
        self.tgt_tok_emb = nn.Embedding(vocab_size, emb_size, padding_idx=pad_token_id)
        self.src_positional_encoding = nn.Parameter(torch.zeros(1, src_max_seq_len, emb_size))  # max length 4096
        self.tgt_positional_encoding = nn.Parameter(torch.zeros(1, tgt_max_seq_len, emb_size))  # max length 4096

        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.generator = nn.Linear(emb_size, vocab_size)
        self.pad_token_id = pad_token_id

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        src_emb = self.src_tok_emb(src) + self.src_positional_encoding[:, :src.size(1), :]
        tgt_emb = self.tgt_tok_emb(tgt) + self.tgt_positional_encoding[:, :tgt.size(1), :]

        output = self.transformer(
            src_emb,
            tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return self.generator(output)

class CharTokenizer:
    """
    Character-level tokenizer for protein sequences (input) and numbers/symbols (output).
    """
    def __init__(self, pad_token="[PAD]", unk_token="[UNK]"):
        protein_letters = list("ACDEFGHIKLMNPQRSTVWYBXZUO")  # common protein chars
        output_symbols = list("0123456789-_*")   # extend as needed
        self.vocab = sorted(set(protein_letters + output_symbols + [pad_token, unk_token]))
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}
        self.pad_token_id = self.stoi[self.pad_token]
        self.unk_token_id = self.stoi[self.unk_token]

    def __len__(self):
        return len(self.vocab)

    def encode(self, text, max_length=None, padding="max_length", truncation=True):
        # Converts a string to a torch tensor of token ids
        ids = [self.stoi.get(ch, self.unk_token_id) for ch in text]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        if padding == "max_length" and max_length is not None:
            ids = ids + [self.pad_token_id] * (max_length - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids, skip_special_tokens=True):
        # Converts a tensor or list of token ids back to string
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        chars = []
        for i in ids:
            ch = self.itos.get(i, self.unk_token)
            if skip_special_tokens and ch in [self.pad_token, self.unk_token]:
                continue
            chars.append(ch)
        return "".join(chars)

    def batch_encode(self, texts, max_length=None, padding="max_length", truncation=True):
        encoded = [self.encode(t, max_length=max_length, padding=padding, truncation=truncation) for t in texts]
        return {"input_ids": encoded}

    def batch_decode(self, batch_ids, skip_special_tokens=True):
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in batch_ids]

    def to_device(self, obj, device="cuda"):
        """
        Move a tensor or list of tensors to the specified device.
        """
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, list):
            return [self.to_device(x, device=device) for x in obj]
        else:
            raise TypeError("Object must be a torch.Tensor or list of torch.Tensors")

tokenizer = CharTokenizer()

model = Seq2SeqTransformer(
    num_encoder_layers=config.get("num_encoder_layers", 6),
    num_decoder_layers=config.get("num_decoder_layers", 6),
    emb_size=config.get("emb_size", 512),
    nhead=config.get("nhead", 8),
    vocab_size=len(tokenizer),
    dim_feedforward=config.get("dim_feedforward", 2048),
    dropout=config.get("dropout", 0.1),
    pad_token_id=tokenizer.pad_token_id if hasattr(tokenizer, "pad_token_id") else 0,
    src_max_seq_len=config.get("src_max_seq_len", 2048),
    tgt_max_seq_len=config.get("tgt_max_seq_len", 128),
)

CACHE_DIR = "transformer_extended_tokenized_ds/"
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
        model_inputs = tokenizer.batch_encode(examples["Sequence"], max_length=config.get("src_max_seq_len", 2048))
        labels = tokenizer.batch_encode(examples["label"], max_length=config.get("tgt_max_seq_len", 128))
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = raw_ds.map(
        preprocess,
        batched=True,
        remove_columns=raw_ds["train"].column_names,
    )

    tokenized.save_to_disk(CACHE_DIR)

def compute_metrics(eval_pred):
    preds = eval_pred.predictions
    labels = eval_pred.label_ids
    preds = np.clip(preds, 0, len(tokenizer) - 1)

    decoded_inputs  = tokenizer.batch_decode(tokenized["validation"]["input_ids"], skip_special_tokens=True)
    decoded_preds  = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

train_dataset = tokenized["train"]
eval_dataset = tokenized["validation"]

train_dataloader = DataLoader(
    train_dataset,
    batch_size=config["per_device_train_batch_size"],
    shuffle=True,
    collate_fn=tokenizer.batch_encode,
)
eval_dataloader = DataLoader(
    eval_dataset,
    batch_size=config["per_device_eval_batch_size"],
    shuffle=False,
    collate_fn=tokenizer.batch_encode,
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=float(config["learning_rate"]),
    weight_decay=float(config["weight_decay"])
)

num_epochs = int(config["num_train_epochs"])
gradient_accumulation_steps = int(config["gradient_accumulation_steps"])
max_grad_norm = float(config["max_grad_norm"])
save_steps = int(config["save_steps"])
eval_steps = int(config["eval_steps"])
logging_steps = int(config["logging_steps"])
output_dir = config["output_dir"]
save_total_limit = int(config["save_total_limit"])

global_step = 0
best_metric = None
saved_checkpoints = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}")
    for step, batch in pbar:
        print(batch)
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss = loss / gradient_accumulation_steps
        loss.backward()
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % logging_steps == 0:
                pbar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})

            if global_step % eval_steps == 0:
                model.eval()
                all_preds = []
                all_labels = []
                with torch.no_grad():
                    for eval_batch in eval_dataloader:
                        eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
                        outputs = model(**eval_batch)
                        if hasattr(outputs, "logits"):
                            preds = torch.argmax(outputs.logits, dim=-1)
                        else:
                            preds = outputs[0]
                        all_preds.append(preds.cpu().numpy())
                        all_labels.append(eval_batch["labels"].cpu().numpy())
                # Flatten predictions and labels
                import numpy as np
                all_preds = np.concatenate(all_preds, axis=0)
                all_labels = np.concatenate(all_labels, axis=0)
                eval_pred = type("EvalPred", (), {"predictions": all_preds, "label_ids": all_labels})()
                metrics = compute_metrics(eval_pred)
                print(f"Step {global_step}: Eval metrics: {metrics}")
                model.train()

            if global_step % save_steps == 0:
                checkpoint_dir = f"{output_dir}/checkpoint-{global_step}"
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                saved_checkpoints.append(checkpoint_dir)
                # Remove older checkpoints if exceeding save_total_limit
                if len(saved_checkpoints) > save_total_limit:
                    to_remove = saved_checkpoints.pop(0)
                    import shutil
                    shutil.rmtree(to_remove, ignore_errors=True)
