import os
import argparse
from enum import Enum, auto
from typing import Set

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


class _GState(Enum):
    """States of the chopping_star output grammar FSM.

    Grammar (simplified):
        output       := domain ( STAR domain )* EOS
        domain       := segments PIPE cath_class
        segments     := segment ( UNDERSCORE segment )*
        segment      := DIGITS DASH DIGITS
        cath_class   := DIGITS ( DOT DIGITS )*
    """
    RANGE_START_DIGIT   = auto()  # first digit(s) of a range start  e.g. "2"
    RANGE_DASH          = auto()  # must emit '-'
    RANGE_END_DIGIT     = auto()  # digit(s) of range end            e.g. "142"
    AFTER_RANGE         = auto()  # can emit ' | '(space-pipe-space), '_', ' * ', or EOS
    SPACE_BEFORE_PIPE   = auto()  # space before '|'
    PIPE                = auto()  # must emit '|'
    SPACE_AFTER_PIPE    = auto()  # space after '|'
    CATH_DIGIT          = auto()  # digit(s) inside a CATH token
    CATH_DOT_OR_END     = auto()  # can emit '.', ' * ', or EOS
    SPACE_BEFORE_STAR   = auto()  # space before '*'
    STAR                = auto()  # must emit '*'
    SPACE_AFTER_STAR    = auto()  # space after '*'
    UNDERSCORE          = auto()  # must emit '_'


def _build_grammar_mask(
    state: _GState,
    token2id: dict,
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Return a boolean mask (vocab_size,) where True = token is ALLOWED."""
    allowed: Set[str] = set()

    digits = set("0123456789")

    if state == _GState.RANGE_START_DIGIT:
        # Digits of range start, then '-' to begin the end number
        allowed = digits | {"-"}

    elif state == _GState.RANGE_DASH:
        allowed = {"-"}

    elif state == _GState.RANGE_END_DIGIT:
        # Digits of range end, then ' ' (leads to '|') or '_' (discontinuous segment)
        allowed = digits | {" ", "_"}

    elif state == _GState.AFTER_RANGE:
        allowed = {" ", "_"}

    elif state == _GState.SPACE_BEFORE_PIPE:
        allowed = {"|"}

    elif state == _GState.PIPE:
        allowed = {" "}

    elif state == _GState.SPACE_AFTER_PIPE:
        # Digits start a CATH class; '-' represents unknown CATH label
        allowed = digits | {"-"}

    elif state == _GState.CATH_DIGIT:
        # More CATH digits, '.' between hierarchy levels, or ' ' before ' * ' / EOS
        allowed = digits | {".", " "}

    elif state == _GState.CATH_DOT_OR_END:
        allowed = {".", " "}

    elif state == _GState.SPACE_BEFORE_STAR:
        allowed = {"*"}

    elif state == _GState.STAR:
        allowed = {" "}

    elif state == _GState.SPACE_AFTER_STAR:
        allowed = digits

    elif state == _GState.UNDERSCORE:
        allowed = digits

    mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    for ch in allowed:
        if ch in token2id:
            mask[token2id[ch]] = True
    return mask


def _next_grammar_state(state: _GState, token: str) -> _GState:
    """Transition the FSM given the emitted token character."""

    digits = set("0123456789")

    if state == _GState.RANGE_START_DIGIT:
        if token in digits:
            return _GState.RANGE_START_DIGIT
        if token == "-":
            return _GState.RANGE_END_DIGIT

    elif state == _GState.RANGE_END_DIGIT:
        if token in digits:
            return _GState.RANGE_END_DIGIT
        if token == " ":
            return _GState.SPACE_BEFORE_PIPE
        if token == "_":
            return _GState.RANGE_START_DIGIT  # discontinuous next segment

    elif state == _GState.SPACE_BEFORE_PIPE:
        if token == "|":
            return _GState.PIPE

    elif state == _GState.PIPE:
        if token == " ":
            return _GState.SPACE_AFTER_PIPE

    elif state == _GState.SPACE_AFTER_PIPE:
        if token in digits:
            return _GState.CATH_DIGIT
        if token == "-":
            # Unknown CATH label represented as '-'; after this only ' * ' or EOS is valid
            return _GState.CATH_DIGIT

    elif state == _GState.CATH_DIGIT:
        if token in digits:
            return _GState.CATH_DIGIT
        if token == ".":
            return _GState.CATH_DIGIT  # next level digit follows
        if token == " ":
            return _GState.SPACE_BEFORE_STAR

    elif state == _GState.SPACE_BEFORE_STAR:
        if token == "*":
            return _GState.STAR

    elif state == _GState.STAR:
        if token == " ":
            return _GState.SPACE_AFTER_STAR

    elif state == _GState.SPACE_AFTER_STAR:
        if token in digits:
            return _GState.RANGE_START_DIGIT

    # Fallback: stay in current state (should not happen with proper masking)
    return state


def grammar_guided_decode(
    model: TextToTextTransformer,
    src: torch.Tensor,
    src_pad_id: int,
    tgt_pad_id: int,
    sos_id: int,
    eos_id: int,
    max_len: int,
    device: torch.device,
    token2id: dict,
    id2token: dict,
) -> torch.Tensor:
    """Greedy decode with grammar-guided token masking.

    At each step only tokens valid under the chopping_star grammar FSM are
    allowed.  This guarantees every generated string is parseable.

    Parameters
    ----------
    token2id / id2token:
        The target tokenizer's vocabulary mappings.
    All other parameters are identical to ``greedy_decode``.
    """
    model.eval()
    src = src.to(device)
    src_key_padding_mask = src.eq(src_pad_id)
    vocab_size = len(token2id)

    bsz = src.size(0)
    tgt = torch.full((bsz, 1), sos_id, dtype=torch.long, device=device)

    # Each sequence in the batch has its own FSM state.
    states = [_GState.RANGE_START_DIGIT] * bsz
    finished = [False] * bsz

    with torch.no_grad():
        for _ in range(max_len):
            tgt_key_padding_mask = tgt.eq(tgt_pad_id)
            logits = model(
                src,
                tgt,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )
            next_logits = logits[:, -1, :].clone()  # (bsz, vocab)

            next_tokens = []
            for b in range(bsz):
                if finished[b]:
                    next_tokens.append(eos_id)
                    continue

                mask = _build_grammar_mask(states[b], token2id, vocab_size, device)
                # Always allow EOS so decoding can terminate.
                eos_tensor_idx = eos_id
                if 0 <= eos_tensor_idx < vocab_size:
                    mask[eos_tensor_idx] = True

                # Apply mask: set disallowed logits to -inf.
                row = next_logits[b].clone()
                row[~mask] = float("-inf")

                chosen = int(row.argmax().item())
                next_tokens.append(chosen)

                if chosen == eos_id:
                    finished[b] = True
                else:
                    ch = id2token.get(chosen, "")
                    states[b] = _next_grammar_state(states[b], ch)

            next_token_tensor = torch.tensor(next_tokens, dtype=torch.long, device=device).unsqueeze(1)
            tgt = torch.cat([tgt, next_token_tensor], dim=1)

            if all(finished):
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

