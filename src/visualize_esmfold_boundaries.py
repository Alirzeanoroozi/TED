"""ESMFold pairwise distance + domain boundary visualisation.

For each selected protein chain:
  1. Run ESMFold to predict a 3-D structure.
  2. Extract Cβ (Cα for Gly) coordinates and compute the NxN Cβ–Cβ
     pairwise distance matrix.
  3. Run the trained TED transformer to predict the chopping_star label.
  4. Plot the distance matrix as a heatmap and overlay predicted domain
     boundaries as horizontal/vertical lines, colour-coded by CATH class.
  5. Save one figure per sequence and a summary grid of all N sequences.

Usage
-----
python visualize_esmfold_boundaries.py \
    --checkpoint transformer_checkpoint.pt \
    --data_parquet_folder data/all_parquet \
    --output_dir esmfold_figures \
    --n_sequences 100 \
    --seed 42

Requirements
------------
    pip install esm  # Meta's ESM library (includes ESMFold)
    pip install matplotlib
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project-local imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset_ import _load_paths          # noqa: E402
from tokenizer_ import TextTokenizer           # noqa: E402
from model import TextToTextTransformer        # noqa: E402
from evaluate import greedy_decode, grammar_guided_decode  # noqa: E402


# ---------------------------------------------------------------------------
# Matplotlib (lazy import so the module can be imported without display)
# ---------------------------------------------------------------------------
def _get_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors
    return plt, mpatches, mcolors


# ---------------------------------------------------------------------------
# Chopping-star parsing helpers
# ---------------------------------------------------------------------------
_DOMAIN_RE = re.compile(
    r"([\d\-_]+)\s*\|\s*([\d.]+|-)"
)


def _parse_chopping_star(chopping_star: str, seq_len: int) -> List[Dict]:
    """Return list of dicts with 'segments' (list of (start,end) 0-based
    half-open) and 'cath' (str or None) for each domain."""
    domains = []
    for raw_domain in chopping_star.split("*"):
        raw_domain = raw_domain.strip()
        if not raw_domain:
            continue
        m = _DOMAIN_RE.match(raw_domain)
        if not m:
            continue
        bounds_text, cath_text = m.group(1).strip(), m.group(2).strip()
        cath = None if cath_text == "-" else cath_text

        segments = []
        for seg in bounds_text.split("_"):
            seg = seg.strip()
            parts = seg.split("-")
            if len(parts) == 2:
                try:
                    # Input is 1-based inclusive → convert to 0-based half-open
                    start = int(parts[0]) - 1
                    end = int(parts[1])
                    if 0 <= start < end <= seq_len:
                        segments.append((start, end))
                except ValueError:
                    pass
        if segments:
            domains.append({"segments": segments, "cath": cath})
    return domains


# ---------------------------------------------------------------------------
# Diversity filtering
# ---------------------------------------------------------------------------
def _is_diverse_candidate(chopping_star: str) -> bool:
    """Return True if chain has multi-domain or non-continuous domains."""
    domains_raw = [d.strip() for d in chopping_star.split("*") if d.strip()]
    if len(domains_raw) > 1:
        return True  # multi-domain
    # single domain: check for discontinuous segments (underscore in bounds)
    for raw in domains_raw:
        if "|" in raw:
            bounds_part = raw.split("|")[0]
            if "_" in bounds_part:
                return True  # non-continuous
    return False


def _select_sequences(
    df,
    n: int,
    seed: int,
) -> "pd.DataFrame":
    """Select up to `n` diverse sequences (multi-domain or non-continuous)
    from `df`, falling back to random sampling if not enough diverse ones."""
    import pandas as pd

    df = df.dropna(subset=["sequence", "chopping_star"]).reset_index(drop=True)

    diverse_mask = df["chopping_star"].astype(str).apply(_is_diverse_candidate)
    diverse_df = df[diverse_mask]
    other_df = df[~diverse_mask]

    rng = np.random.default_rng(seed)

    n_diverse = min(len(diverse_df), n)
    chosen_diverse = diverse_df.sample(n=n_diverse, random_state=seed) if n_diverse > 0 else pd.DataFrame()

    n_remaining = n - n_diverse
    if n_remaining > 0 and len(other_df) > 0:
        n_remaining = min(n_remaining, len(other_df))
        chosen_other = other_df.sample(n=n_remaining, random_state=seed)
    else:
        chosen_other = pd.DataFrame()

    selected = pd.concat([chosen_diverse, chosen_other], ignore_index=True)
    return selected.reset_index(drop=True)


# ---------------------------------------------------------------------------
# ESMFold Cβ distance matrix
# ---------------------------------------------------------------------------
def _load_esmfold():
    """Load ESMFold model (downloads weights on first call, ~2.5 GB)."""
    try:
        import esm
    except ImportError as exc:
        raise ImportError(
            "ESMFold requires the 'esm' package. Install with:\n"
            "    pip install esm\n"
            "or follow https://github.com/facebookresearch/esm for CUDA builds."
        ) from exc
    print("Loading ESMFold model (this may take a few minutes on first run)…")
    model = esm.pretrained.esmfold_v1()
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def _esmfold_cb_distance_matrix(
    esmfold_model,
    sequence: str,
) -> np.ndarray:
    """Run ESMFold on `sequence` and return the NxN Cβ–Cβ distance matrix.

    Cα is used instead of Cβ for glycine residues.
    Returns an (N, N) float32 numpy array.
    """
    device = next(esmfold_model.parameters()).device

    with torch.no_grad():
        output = esmfold_model.infer_pdb(sequence)

    # Parse PDB string to extract Cβ / Cα coordinates.
    coords = _parse_cb_coords_from_pdb(output, len(sequence))
    coords = np.array(coords, dtype=np.float32)  # (N, 3)

    diff = coords[:, None, :] - coords[None, :, :]       # (N, N, 3)
    dist_matrix = np.sqrt((diff ** 2).sum(axis=-1))      # (N, N)
    return dist_matrix


def _parse_cb_coords_from_pdb(pdb_string: str, n_residues: int) -> List[Tuple[float, float, float]]:
    """Extract one coordinate per residue from a PDB string.

    Priority: CB atom; fallback to CA (for Gly or missing CB).
    Returns list of (x, y, z) tuples, one per residue position (1-indexed).
    """
    cb_coords: Dict[int, Tuple[float, float, float]] = {}
    ca_coords: Dict[int, Tuple[float, float, float]] = {}

    for line in pdb_string.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        res_seq = int(line[22:26].strip())
        x = float(line[30:38].strip())
        y = float(line[38:46].strip())
        z = float(line[46:54].strip())

        if atom_name == "CB":
            cb_coords[res_seq] = (x, y, z)
        elif atom_name == "CA":
            ca_coords[res_seq] = (x, y, z)

    coords = []
    for res_idx in range(1, n_residues + 1):
        if res_idx in cb_coords:
            coords.append(cb_coords[res_idx])
        elif res_idx in ca_coords:
            coords.append(ca_coords[res_idx])
        else:
            # Missing residue: use NaN placeholder (rare in ESMFold output)
            coords.append((float("nan"), float("nan"), float("nan")))
    return coords


# ---------------------------------------------------------------------------
# TED model prediction
# ---------------------------------------------------------------------------
def _load_ted_model(checkpoint_path: str, device: torch.device):
    """Load the TED transformer and tokenizers from a .pt checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_args = ckpt.get("args", {})

    # Re-fit tokenizers from a small data slice to rebuild the vocabulary.
    # The checkpoint stores src/tgt vocab sizes but not the actual mappings,
    # so we need the original parquet data.
    return ckpt, ckpt_args


def _predict_chopping_star(
    model: TextToTextTransformer,
    src_tokenizer: TextTokenizer,
    tgt_tokenizer: TextTokenizer,
    sequence: str,
    max_src_len: int,
    max_tgt_len: int,
    device: torch.device,
    use_grammar_guided: bool = False,
) -> str:
    """Run the TED model on `sequence` and return the predicted chopping_star."""
    sequence = sequence[:max_src_len]
    src_ids = src_tokenizer.encode(sequence, add_sos_eos=True)
    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)

    if use_grammar_guided:
        generated_ids = grammar_guided_decode(
            model,
            src_tensor,
            src_pad_id=src_tokenizer.pad_id,
            tgt_pad_id=tgt_tokenizer.pad_id,
            sos_id=tgt_tokenizer.sos_id,
            eos_id=tgt_tokenizer.eos_id,
            max_len=max_tgt_len,
            device=device,
            token2id=tgt_tokenizer.token2id,
            id2token=tgt_tokenizer.id2token,
        )
    else:
        generated_ids = greedy_decode(
            model,
            src_tensor,
            src_pad_id=src_tokenizer.pad_id,
            tgt_pad_id=tgt_tokenizer.pad_id,
            sos_id=tgt_tokenizer.sos_id,
            eos_id=tgt_tokenizer.eos_id,
            max_len=max_tgt_len,
            device=device,
        )

    pred_text = tgt_tokenizer.decode(generated_ids[0].tolist(), strip_special=True)
    return pred_text


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
_CMAP_DIST = "viridis_r"

# Colour cycle for domain overlays (up to 10 distinct colours)
_DOMAIN_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
]


def _boundary_positions(domains: List[Dict]) -> List[int]:
    """Return sorted list of unique boundary residue indices (0-based half-open)."""
    boundaries = set()
    for d in domains:
        for start, end in d["segments"]:
            boundaries.add(start)
            boundaries.add(end)
    return sorted(boundaries)


def _plot_single(
    dist_matrix: np.ndarray,
    pred_domains: List[Dict],
    gt_domains: Optional[List[Dict]],
    sequence_id: str,
    output_path: Path,
):
    """Save a single distance-matrix figure with domain boundary overlays."""
    plt, mpatches, mcolors = _get_matplotlib()

    n = dist_matrix.shape[0]
    fig, axes = plt.subplots(
        1, 2 if gt_domains is not None else 1,
        figsize=(10 if gt_domains is not None else 6, 5),
        squeeze=False,
    )

    def _draw_panel(ax, domains, title):
        ax.imshow(dist_matrix, cmap=_CMAP_DIST, aspect="auto", origin="upper")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Residue index")
        ax.set_ylabel("Residue index")

        legend_patches = []
        for dom_idx, dom in enumerate(domains):
            color = _DOMAIN_COLORS[dom_idx % len(_DOMAIN_COLORS)]
            cath_label = dom.get("cath") or "unknown"
            for seg_start, seg_end in dom["segments"]:
                for pos in (seg_start, seg_end):
                    if 0 < pos < n:
                        ax.axvline(x=pos - 0.5, color=color, linewidth=1.0, alpha=0.85)
                        ax.axhline(y=pos - 0.5, color=color, linewidth=1.0, alpha=0.85)
            patch = mpatches.Patch(color=color, label=f"D{dom_idx + 1}: {cath_label}")
            legend_patches.append(patch)

        if legend_patches:
            ax.legend(handles=legend_patches, fontsize=6, loc="upper right",
                      framealpha=0.7, borderpad=0.5)

    _draw_panel(axes[0, 0], pred_domains, f"Predicted  ({sequence_id})")
    if gt_domains is not None:
        _draw_panel(axes[0, 1], gt_domains, f"Ground truth  ({sequence_id})")

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_summary_grid(
    figure_paths: List[Path],
    output_path: Path,
    cols: int = 5,
):
    """Assemble all individual figures into a single summary grid image."""
    plt, _, _ = _get_matplotlib()
    from PIL import Image  # only needed for grid assembly

    images = [Image.open(str(p)) for p in figure_paths if p.exists()]
    if not images:
        return

    n = len(images)
    rows = math.ceil(n / cols)
    w, h = images[0].size
    grid = Image.new("RGB", (cols * w, rows * h), color=(255, 255, 255))

    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        grid.paste(img.resize((w, h)), (c * w, r * h))

    grid.save(str(output_path))
    print(f"Saved summary grid → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Visualise ESMFold Cβ–Cβ distance maps with TED domain boundary overlays."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="transformer_checkpoint.pt",
        help="Path to TED .pt checkpoint saved by train_lightning.py.",
    )
    parser.add_argument(
        "--data_parquet_folder",
        type=str,
        required=True,
        help="Folder containing parquet shards (used to build tokenizers and select sequences).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="esmfold_figures",
        help="Directory where per-sequence figures and the summary grid are saved.",
    )
    parser.add_argument(
        "--n_sequences",
        type=int,
        default=100,
        help="Number of sequences to visualise (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sequence selection.",
    )
    parser.add_argument(
        "--show_gt",
        action="store_true",
        help="Show ground-truth boundaries side-by-side with predictions.",
    )
    parser.add_argument(
        "--grammar_guided_decoding",
        action="store_true",
        help="Use grammar-guided FSM decoding for predictions.",
    )
    parser.add_argument(
        "--grid_cols",
        type=int,
        default=5,
        help="Number of columns in the summary grid image (default: 5).",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=None,
        help="Skip sequences longer than this (ESMFold is slow on very long chains).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load parquet data ──────────────────────────────────────────────────
    parquet_folder = Path(args.data_parquet_folder).expanduser()
    if parquet_folder.is_dir():
        parquet_files = sorted(str(p) for p in parquet_folder.glob("*.parquet"))
    else:
        parquet_files = [str(parquet_folder)]

    if not parquet_files:
        raise RuntimeError(f"No .parquet files found under {parquet_folder}")

    print(f"Loading data from {len(parquet_files)} parquet file(s)…")
    df = _load_paths(parquet_files)

    if args.max_seq_len is not None:
        df = df[df["sequence"].str.len() <= args.max_seq_len].reset_index(drop=True)
        print(f"After max_seq_len={args.max_seq_len} filter: {len(df)} sequences remain.")

    selected_df = _select_sequences(df, args.n_sequences, args.seed)
    print(f"Selected {len(selected_df)} sequences ({sum(selected_df['chopping_star'].astype(str).apply(_is_diverse_candidate))} diverse).")

    # ── Build tokenizers ──────────────────────────────────────────────────
    fit_df = _load_paths(parquet_files[:3] if len(parquet_files) >= 3 else parquet_files)
    src_tokenizer = TextTokenizer().fit(fit_df["sequence"].astype(str).tolist())
    tgt_tokenizer = TextTokenizer().fit(fit_df["chopping_star"].astype(str).tolist())

    # =========================================================================
    # PHASE 1 — TED inference (runs first, then GPU memory is fully released)
    # Both models cannot coexist in GPU memory, so we run TED, collect all
    # predictions, then delete it before loading ESMFold.
    # =========================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})

    max_src_len = ckpt_args.get("max_src_len", 1024)
    max_tgt_len = ckpt_args.get("max_tgt_len", 256)

    ted_model = TextToTextTransformer(
        src_vocab_size=ckpt["src_vocab_size"],
        tgt_vocab_size=ckpt["tgt_vocab_size"],
        d_model=ckpt_args.get("d_model", 256),
        nhead=ckpt_args.get("nhead", 8),
        num_encoder_layers=ckpt_args.get("num_encoder_layers", 4),
        num_decoder_layers=ckpt_args.get("num_decoder_layers", 4),
        dim_feedforward=ckpt_args.get("dim_feedforward", 1024),
        dropout=ckpt_args.get("dropout", 0.1),
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
    ).to(device)
    ted_model.load_state_dict(ckpt["model_state_dict"])
    ted_model.eval()

    print(f"\n── Phase 1: TED inference on {len(selected_df)} sequences ──")
    inference_results: List[Dict] = []
    for idx, row in selected_df.iterrows():
        sequence = str(row["sequence"]).strip()
        gt_chopping = str(row.get("chopping_star", "")).strip()
        seq_id = str(row.get("uniprot_id", row.get("id", f"seq{idx}"))).strip()
        print(f"  [{idx + 1}/{len(selected_df)}] inferring {seq_id}  len={len(sequence)}")
        try:
            pred_chopping = _predict_chopping_star(
                ted_model, src_tokenizer, tgt_tokenizer,
                sequence, max_src_len, max_tgt_len, device,
                use_grammar_guided=args.grammar_guided_decoding,
            )
        except Exception as exc:
            print(f"    TED inference ERROR on {seq_id}: {exc}")
            pred_chopping = ""
        inference_results.append({
            "seq_id": seq_id,
            "sequence": sequence,
            "gt_chopping": gt_chopping,
            "pred_chopping": pred_chopping,
        })

    # Explicitly free TED model from GPU before loading ESMFold.
    del ted_model, ckpt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("TED model unloaded from GPU.\n")

    # =========================================================================
    # PHASE 2 — ESMFold distograms + visualisation
    # ESMFold is loaded only after TED has been removed from GPU memory.
    # =========================================================================
    print("── Phase 2: ESMFold distograms + plotting ──")
    esmfold = _load_esmfold()

    figure_paths: List[Path] = []
    failed = 0

    for entry in inference_results:
        seq_id = entry["seq_id"]
        sequence = entry["sequence"]
        gt_chopping = entry["gt_chopping"]
        pred_chopping = entry["pred_chopping"]
        print(f"  ESMFold + plot: {seq_id}  len={len(sequence)}")

        try:
            dist_matrix = _esmfold_cb_distance_matrix(esmfold, sequence)

            pred_domains = _parse_chopping_star(pred_chopping, len(sequence))
            gt_domains = _parse_chopping_star(gt_chopping, len(sequence)) if args.show_gt else None

            fig_path = output_dir / f"{seq_id.replace('/', '_')}.png"
            _plot_single(dist_matrix, pred_domains, gt_domains, seq_id, fig_path)
            figure_paths.append(fig_path)
            print(f"    Saved → {fig_path.name}  pred='{pred_chopping[:60]}…'")

        except Exception as exc:
            print(f"    ERROR on {seq_id}: {exc}")
            failed += 1
            continue

    print(f"\nDone. {len(figure_paths)} figures saved, {failed} failed.")

    # ── Summary grid ──────────────────────────────────────────────────────
    if figure_paths:
        try:
            grid_path = output_dir / "summary_grid.png"
            _plot_summary_grid(figure_paths, grid_path, cols=args.grid_cols)
        except ImportError:
            print("Pillow not installed — skipping summary grid. Install with: pip install Pillow")


if __name__ == "__main__":
    main()
