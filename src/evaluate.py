import os
import argparse

import torch

from data.dataset_ import _load_paths
from tokenizer_ import TextTokenizer
from model import TextToTextTransformer


def greedy_decode(
    model: TextToTextTransformer,
    src: torch.Tensor,
    src_pad_id: int,
    tgt_pad_id: int,
    sos_id: int,
    eos_id: int,
    max_len: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    src = src.to(device)
    src_key_padding_mask = src.eq(src_pad_id)

    bsz = src.size(0)
    tgt = torch.full(
        (bsz, 1),
        sos_id,
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        for _ in range(max_len):
            tgt_key_padding_mask = tgt.eq(tgt_pad_id)
            logits = model(
                src,
                tgt,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_token], dim=1)
            if (next_token == eos_id).all():
                break

    return tgt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="transformer_checkpoint.pt",
        help="Path to trained checkpoint (.pt) saved by train_lightning.py.",
    )
    parser.add_argument(
        "--data_parquet_folder",
        type=str,
        default=None,
        help="Folder with parquet shards. If None, use path stored in checkpoint args.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of sequences to generate for inspection.",
    )
    parser.add_argument(
        "--max_gen_len",
        type=int,
        default=None,
        help="Maximum generated target length (defaults to checkpoint max_tgt_len).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})

    data_parquet_folder = (
        args.data_parquet_folder
        if args.data_parquet_folder is not None
        else ckpt_args.get("data_parquet_folder", "data/parquet_sequences")
    )

    parquet_files = [
        os.path.join(data_parquet_folder, f)
        for f in os.listdir(data_parquet_folder)
        if f.endswith(".parquet")
    ]
    parquet_files = sorted(parquet_files)
    if not parquet_files:
        raise RuntimeError(f"No .parquet files found under {data_parquet_folder}")

    df = _load_paths(parquet_files[:3] if len(parquet_files) >= 3 else parquet_files)

    src_tokenizer = TextTokenizer().fit(df["sequence"].astype(str).tolist())
    tgt_tokenizer = TextTokenizer().fit(df["chopping_star"].astype(str).tolist())

    src_vocab_size = ckpt["src_vocab_size"]
    tgt_vocab_size = ckpt["tgt_vocab_size"]

    d_model = ckpt_args.get("d_model", 256)
    nhead = ckpt_args.get("nhead", 8)
    num_encoder_layers = ckpt_args.get("num_encoder_layers", 4)
    num_decoder_layers = ckpt_args.get("num_decoder_layers", 4)
    dim_feedforward = ckpt_args.get("dim_feedforward", 1024)
    dropout = ckpt_args.get("dropout", 0.1)
    max_src_len = ckpt_args.get("max_src_len", 1024)
    max_tgt_len = ckpt_args.get("max_tgt_len", 128)

    if args.max_gen_len is not None:
        max_tgt_len = args.max_gen_len

    model = TextToTextTransformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])

    sample_df = df.sample(
        n=min(args.num_samples, len(df)),
        random_state=0,
    ).reset_index(drop=True)

    for i, row in sample_df.iterrows():
        src_text = str(row["sequence"])[:max_src_len]
        tgt_true = str(row.get("chopping_star", "")) if "chopping_star" in row else ""

        src_ids = src_tokenizer.encode(src_text, add_sos_eos=True)
        src_tensor = torch.tensor(
            src_ids,
            dtype=torch.long,
        ).unsqueeze(0)

        generated_ids = greedy_decode(
            model,
            src_tensor,
            src_pad_id=src_tokenizer.pad_id,
            tgt_pad_id=tgt_tokenizer.pad_id,
            sos_id=tgt_tokenizer.sos_id,
            eos_id=tgt_tokenizer.eos_id,
            max_len=max_tgt_len,
            device=device,
        )[0].tolist()

        pred_text = tgt_tokenizer.decode(generated_ids, strip_special=True)

        print("=" * 80)
        print(f"[{i}] SEQUENCE:")
        print(src_text)
        print("\nTARGET (chopping_star):")
        print(tgt_true)
        print("\nPREDICTION:")
        print(pred_text)


if __name__ == "__main__":
    main()

