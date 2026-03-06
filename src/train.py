import os
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import pandas as pd
import wandb
from dotenv import load_dotenv
from dataset_ import get_dataloaders
from tokenizer_ import TextTokenizer
from model import TextToTextTransformer

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        src = batch["src"].to(device, non_blocking=True)
        tgt_in = batch["tgt_in"].to(device, non_blocking=True)
        tgt_out = batch["tgt_out"].to(device, non_blocking=True)

        src_key_padding_mask = src == loader.dataset.src_tokenizer.pad_id
        tgt_key_padding_mask = tgt_in == loader.dataset.tgt_tokenizer.pad_id

        logits = model(
            src,
            tgt_in,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        # (B, T, V) -> (B*T, V), (B*T); ignore padding via criterion ignore_index
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / n_batches
    return avg_loss

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        src = batch["src"].to(device, non_blocking=True)
        tgt_in = batch["tgt_in"].to(device, non_blocking=True)
        tgt_out = batch["tgt_out"].to(device, non_blocking=True)

        src_key_padding_mask = src == loader.dataset.src_tokenizer.pad_id
        tgt_key_padding_mask = tgt_in == loader.dataset.tgt_tokenizer.pad_id

        logits = model(
            src,
            tgt_in,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / n_batches
    return avg_loss

def main_worker(rank, world_size, args):
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    # Build tokenizers from data
    df = pd.read_csv(args.data_csv)
    df = df.dropna(subset=["sequence", "chopping_star"])
    src_tok = TextTokenizer().fit(df["sequence"].astype(str).tolist())
    tgt_tok = TextTokenizer().fit(df["chopping_star"].astype(str).tolist())

    loader = get_dataloaders(
        args.data_csv,
        src_tok,
        tgt_tok,
        batch_size=args.batch_size,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        world_size=world_size,
        rank=rank,
    )

    model = TextToTextTransformer(
        src_vocab_size=src_tok.vocab_size,
        tgt_vocab_size=tgt_tok.vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
    ).to(device)
    model = DDP(model, device_ids=[rank])

    # Init wandb only on rank 0 so we get a single run (not one per GPU)
    if rank == 0:
        load_dotenv()

        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(
            project="ted-transformer",
            name="ted-transformer",
            config=vars(args),
        )

    criterion = nn.CrossEntropyLoss(ignore_index=tgt_tok.pad_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        loader.sampler.set_epoch(epoch)
        train_loss = train_one_epoch(model, loader, criterion, optimizer, device)
        eval_loss = evaluate(model, loader, criterion, device)
        print(f"Rank {rank} Epoch {epoch} train loss: {train_loss:.4f}, eval loss: {eval_loss:.4f}")
        if rank == 0:
            wandb.log({"train_loss": train_loss, "eval_loss": eval_loss, "epoch": epoch})

    if rank == 0 and args.save_path:
        torch.save(
            {
                "model_state_dict": model.module.state_dict(),
                "src_vocab_size": src_tok.vocab_size,
                "tgt_vocab_size": tgt_tok.vocab_size,
                "args": vars(args),
            },
            args.save_path,
        )
        print(f"Saved checkpoint to {args.save_path}")
    if rank == 0:
        wandb.finish()

    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, default="data/data.csv", help="Path to data.csv")
    parser.add_argument("--batch_size", type=int, default=8, help="Per-GPU batch size")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_encoder_layers", type=int, default=4)
    parser.add_argument("--num_decoder_layers", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_src_len", type=int, default=512)
    parser.add_argument("--max_tgt_len", type=int, default=256)
    parser.add_argument("--save_path", type=str, default="transformer_checkpoint.pt")
    parser.add_argument("--master_addr", type=str, default="localhost")
    parser.add_argument("--master_port", type=str, default="29500")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("This script is intended for 2 GPUs. Running on 1 GPU with DDP (or use torchrun with 2 GPUs).")
        world_size = max(1, world_size)

    torch.multiprocessing.spawn(
        main_worker,
        args=(world_size, args),
        nprocs=world_size,
        join=True,
    )

if __name__ == "__main__":
    main()
