"""Lightning training for the two-stage TEDPred model.

Mirrors the structure of ``src/train_lightning.py`` (W&B logging, periodic
benchmark evaluation with the *same* metric keys so existing dashboards keep
working) but trains the decomposed ESM2 + segmentation + hierarchical-CATH model
with warmup+cosine LR, label smoothing, and length-bucketed batching.

Run (from repo root, 'ted' env):
    python src/two_stage/train.py --data_parquet_folder data/all_parquet \
        --esm_model_name esm2_t33_650M_UR50D --epochs 30 --batch_size 4
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# --- path bootstrap: make `two_stage`, `data`, `benchmark` importable -------- #
SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_DIR.parent
for p in (str(SRC_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from dotenv import load_dotenv
import wandb

from data.dataset_ import _load_paths
from two_stage.cath_vocab import CathVocab
from two_stage.dataset import TwoStageDataset, collate_fn, LengthBucketedSampler
from two_stage.model import TwoStageDomainModel
from two_stage import losses

try:
    from benchmark.ted_eval import EvalConfig, evaluate_target
except ImportError:  # pragma: no cover
    from ted_eval import EvalConfig, evaluate_target


_CATH_RE = re.compile(r"\|\s*([\d.]+)")


def resolve_parquet_inputs(data_path: str) -> list[str]:
    path = Path(data_path).expanduser()
    path = path if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    if path.is_file():
        return [str(path)]
    if path.is_dir():
        files = sorted(str(f) for f in path.glob("*.parquet"))
        if not files:
            raise RuntimeError(f"No .parquet files under {path}")
        return files
    raise FileNotFoundError(f"Data path does not exist: {path}")


def extract_cath_codes(chopping_stars: list[str]) -> list[str]:
    codes: list[str] = []
    for cs in chopping_stars:
        codes.extend(_CATH_RE.findall(str(cs)))
    return codes


# --------------------------------------------------------------------------- #
# DataModule
# --------------------------------------------------------------------------- #
class TwoStageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_parquet_folder: str,
        batch_size: int = 4,
        max_len: int = 1022,
        num_workers: int = 0,
        cath_min_count: int = 5,
        benchmark_eval_subset_size: int = 100,
        benchmark_eval_seed: int = 42,
        vocab_save_path: str | None = None,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        shuffle_seed: int = 42,
    ):
        super().__init__()
        self.data_parquet_folder = data_parquet_folder
        self.batch_size = batch_size
        self.max_len = max_len
        self.num_workers = num_workers
        self.cath_min_count = cath_min_count
        self.benchmark_eval_subset_size = benchmark_eval_subset_size
        self.benchmark_eval_seed = benchmark_eval_seed
        self.vocab_save_path = vocab_save_path
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.shuffle_seed = shuffle_seed
        self.cath_vocab: CathVocab | None = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.benchmark_val_subset = None
        self._train_lengths: list[int] | None = None

    def setup(self, stage: str | None = None):
        if self.train_dataset is not None:
            return
        files = resolve_parquet_inputs(self.data_parquet_folder)
        df = _load_paths(files)
        df = df.sample(frac=1, random_state=self.shuffle_seed).reset_index(drop=True)
        n = len(df)
        n_val = max(1, int(n * self.val_ratio))
        n_test = max(1, int(n * self.test_ratio))
        n_train = max(1, n - n_val - n_test)
        train_df = df.iloc[:n_train].reset_index(drop=True)
        val_df = df.iloc[n_train : n_train + n_val].reset_index(drop=True)
        test_df = df.iloc[n_train + n_val :].reset_index(drop=True)

        # Build the CATH vocab from TRAIN labels only (no leakage).
        codes = extract_cath_codes(train_df["chopping_star"].astype(str).tolist())
        self.cath_vocab = CathVocab.build(codes, min_count=self.cath_min_count)
        rank_zero_info(f"CATH vocab level sizes: {self.cath_vocab.level_sizes}")
        if self.vocab_save_path:
            Path(self.vocab_save_path).parent.mkdir(parents=True, exist_ok=True)
            self.cath_vocab.save(self.vocab_save_path)
            rank_zero_info(f"Saved CATH vocab -> {self.vocab_save_path}")

        self.train_dataset = TwoStageDataset(train_df, self.cath_vocab, self.max_len)
        self.val_dataset = TwoStageDataset(val_df, self.cath_vocab, self.max_len)
        self.test_dataset = TwoStageDataset(test_df, self.cath_vocab, self.max_len)

        self._train_lengths = (
            train_df["sequence"].astype(str).str.len().clip(upper=self.max_len).tolist()
        )

        if len(self.val_dataset) > 0:
            size = min(self.benchmark_eval_subset_size, len(self.val_dataset))
            g = torch.Generator().manual_seed(self.benchmark_eval_seed)
            idx = torch.randperm(len(self.val_dataset), generator=g)[:size].tolist()
            self.benchmark_val_subset = Subset(self.val_dataset, idx)

    def train_dataloader(self):
        sampler = LengthBucketedSampler(
            self._train_lengths, batch_size=self.batch_size, shuffle=True,
            seed=self.shuffle_seed,
        )
        return DataLoader(
            self.train_dataset, batch_sampler=sampler, collate_fn=collate_fn,
            num_workers=self.num_workers, pin_memory=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=self.num_workers,
        )

    def benchmark_val_dataloader(self):
        if not self.benchmark_val_subset:
            return None
        return DataLoader(
            self.benchmark_val_subset, batch_size=self.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=self.num_workers,
        )


# --------------------------------------------------------------------------- #
# LightningModule
# --------------------------------------------------------------------------- #
class TwoStageLightningModule(pl.LightningModule):
    def __init__(
        self,
        cath_vocab: CathVocab,
        esm_model_name: str = "esm2_t33_650M_UR50D",
        d_model: int = 512,
        nhead: int = 8,
        num_seg_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pair_dim: int = 128,
        cath_hidden: int = 512,
        cath_cond_dim: int = 64,
        esm_dtype: str = "bf16",
        esm_chunk_size: int = 1022,
        esm_chunk_overlap: int = 256,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        label_smoothing: float = 0.1,
        residue_loss_weight: float = 1.0,
        pair_loss_weight: float = 1.0,
        cath_loss_weight: float = 1.0,
        max_pair_len: int = 768,
        cluster_method: str = "connected_components",
        min_domain_len: int = 20,
        benchmark_num_logged_samples: int = 5,
        # ---- Tier 3: CATH long-tail handling (CATH head only) ----
        cath_loss_type: str = "ce",          # ce | focal | class_balanced | cb_focal
        focal_gamma: float = 2.0,
        cath_level_weights: tuple = (1.0, 1.0, 1.0, 1.0),
        cath_curriculum_steps: int = 0,      # >0 ramps deeper levels in over N steps
        cath_class_weights: list | None = None,  # per-level tensors (class-balanced)
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["cath_vocab", "cath_class_weights"])
        self.cath_vocab = cath_vocab
        self.model = TwoStageDomainModel(
            cath_vocab=cath_vocab,
            esm_model_name=esm_model_name,
            d_model=d_model, nhead=nhead, num_seg_layers=num_seg_layers,
            dim_feedforward=dim_feedforward, dropout=dropout, pair_dim=pair_dim,
            cath_hidden=cath_hidden, cath_cond_dim=cath_cond_dim,
            esm_chunk_size=esm_chunk_size, esm_chunk_overlap=esm_chunk_overlap,
            esm_dtype=esm_dtype,
        )
        self.benchmark_eval_config = EvalConfig(input_indexing="one_based_inclusive")
        self._pair_gen = torch.Generator()

        # Register class-balanced weights as buffers so they follow the module to
        # the GPU. Used only when cath_loss_type in {class_balanced, cb_focal}.
        self._n_cath_levels = len(cath_vocab.level_sizes)
        if cath_class_weights is not None:
            for lvl, w in enumerate(cath_class_weights):
                self.register_buffer(f"cath_cw_{lvl}", torch.as_tensor(w, dtype=torch.float))
            self._has_class_weights = True
        else:
            self._has_class_weights = False

    def _class_weights_list(self):
        if not self._has_class_weights:
            return None
        return [getattr(self, f"cath_cw_{lvl}") for lvl in range(self._n_cath_levels)]

    def _current_level_weights(self):
        """Static CATH level weights, optionally ramping deeper levels in (3.2)."""
        base = list(self.hparams.cath_level_weights)
        steps = int(self.hparams.cath_curriculum_steps)
        if steps <= 0:
            return base
        seg = steps / len(base)
        gstep = float(self.global_step)
        out = []
        for lvl in range(len(base)):
            f = min(1.0, max(0.0, (gstep - lvl * seg) / max(1.0, seg)))
            out.append(base[lvl] * f)
        out[0] = base[0]  # coarse level always fully active
        return out

    # ---- backbone device management ------------------------------------ #
    def _ensure_backbone(self):
        if (not self.model.backbone._ready) or (self.model.backbone.device != self.device):
            self.model.prepare_backbone(self.device)

    def on_fit_start(self):
        self._ensure_backbone()

    def on_validation_start(self):
        self._ensure_backbone()

    # ---- shared step ---------------------------------------------------- #
    def _compute_losses(self, batch):
        self._ensure_backbone()
        emb, mask = self.model.embed(batch["sequences"])  # backbone, no_grad
        assign = batch["assign"].to(self.device)
        mask = mask.to(self.device)
        domains = [(b, rm.to(self.device)) for (b, rm) in batch["domains"]]
        parent_ids = batch["cath_targets"].to(self.device)

        out = self.model.forward_train(emb, mask, domains, parent_ids)

        res_loss = losses.residue_bce_loss(out["residue_logit"], assign, mask)
        pair_loss = losses.pairwise_bce_loss(
            out["pair_logit"], assign, mask,
            max_pair_len=self.hparams.max_pair_len, generator=self._pair_gen,
        )
        if out["cath_logits"] is not None and parent_ids.numel() > 0:
            loss_type = self.hparams.cath_loss_type
            use_focal = loss_type in ("focal", "cb_focal")
            use_cw = loss_type in ("class_balanced", "cb_focal")
            cath_loss = losses.cath_loss(
                out["cath_logits"], parent_ids,
                label_smoothing=0.0 if use_focal else self.hparams.label_smoothing,
                focal_gamma=self.hparams.focal_gamma if use_focal else 0.0,
                class_weights=self._class_weights_list() if use_cw else None,
                level_weights=self._current_level_weights(),
            )
        else:
            cath_loss = torch.zeros((), device=self.device)

        total = (
            self.hparams.residue_loss_weight * res_loss
            + self.hparams.pair_loss_weight * pair_loss
            + self.hparams.cath_loss_weight * cath_loss
        )
        return total, {"res": res_loss, "pair": pair_loss, "cath": cath_loss}

    def training_step(self, batch, batch_idx):
        total, parts = self._compute_losses(batch)
        self.log("train_loss", total, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_res_loss", parts["res"], on_step=True, on_epoch=False)
        self.log("train_pair_loss", parts["pair"], on_step=True, on_epoch=False)
        self.log("train_cath_loss", parts["cath"], on_step=True, on_epoch=False)
        return total

    def validation_step(self, batch, batch_idx):
        total, parts = self._compute_losses(batch)
        self.log("eval_loss", total, on_step=False, on_epoch=True, sync_dist=True)
        self.log("eval_res_loss", parts["res"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("eval_pair_loss", parts["pair"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("eval_cath_loss", parts["cath"], on_step=False, on_epoch=True, sync_dist=True)
        return total

    # ---- benchmark evaluation (same metric keys as the baseline) -------- #
    @torch.no_grad()
    def run_benchmark_evaluation(self, dataloader=None):
        dm = self.trainer.datamodule
        dataloader = dataloader or dm.benchmark_val_dataloader()
        if dataloader is None:
            return None
        self._ensure_backbone()
        was_training = self.training
        self.eval()

        keys = ["iou_chain", "ndo", "correct_prop", "correct_cath",
                "cath_level_score", "boundary_distance_score"]
        rows = []          # one dict per chain (metrics + gt_domain_count + parse flags)
        sample_rows = []
        max_logged = max(0, int(self.hparams.benchmark_num_logged_samples))

        for batch in dataloader:
            preds = self.model.predict_chopping_star(
                batch["sequences"],
                cluster_method=self.hparams.cluster_method,
                min_domain_len=self.hparams.min_domain_len,
            )
            for seq, tgt, pred in zip(batch["sequences"], batch["chopping_star"], preds):
                m = evaluate_target(tgt, pred, nres=len(seq), sequence=seq,
                                    config=self.benchmark_eval_config)
                row = {k: float(m[k]) for k in keys}
                row["pred_parse_ok"] = int(bool(m["pred_parse_ok"]))
                row["domain_count_match"] = int(bool(m["domain_count_match"]))
                row["gt_domain_count"] = int(m["gt_domain_count"])
                rows.append(row)
                if len(sample_rows) < max_logged:
                    sample_rows.append({
                        "input_len": len(seq), "target": tgt, "predicted": pred,
                        "iou_chain": float(m["iou_chain"]),
                        "correct_cath": float(m["correct_cath"]),
                        "cath_level_score": float(m["cath_level_score"]),
                        "pred_domain_count": int(m["pred_domain_count"]),
                        "gt_domain_count": int(m["gt_domain_count"]),
                    })

        if was_training:
            self.train()
        if not rows:
            return None

        def mean(x):
            return float(np.nanmean(x)) if x else float("nan")

        # Break out by domain count so multi-domain progress (the bottleneck) is
        # visible during training, not just the aggregate.
        groups = {
            "": rows,
            "single_": [r for r in rows if r["gt_domain_count"] == 1],
            "multi_": [r for r in rows if r["gt_domain_count"] >= 2],
        }
        out = {}
        for prefix, grp in groups.items():
            if not grp:
                continue
            out[f"val_subset_{prefix}n"] = len(grp)
            for k in keys:
                out[f"val_subset_{prefix}{k}"] = mean([r[k] for r in grp])
            out[f"val_subset_{prefix}pred_parse_ok_rate"] = mean([r["pred_parse_ok"] for r in grp])
            out[f"val_subset_{prefix}domain_count_match_rate"] = mean([r["domain_count_match"] for r in grp])

        if sample_rows and isinstance(self.logger, WandbLogger):
            cols = list(sample_rows[0].keys())
            table = wandb.Table(columns=cols)
            for r in sample_rows:
                table.add_data(*[r[c] for c in cols])
            out["benchmark_samples_table"] = table
        return out

    # ---- optimizer + warmup/cosine schedule ---------------------------- #
    def configure_optimizers(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        try:
            max_steps = int(self.trainer.estimated_stepping_batches)
        except Exception:
            max_steps = 100000
        warmup = max(1, int(self.hparams.warmup_steps))

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / warmup
            progress = (step - warmup) / max(1, max_steps - warmup)
            progress = min(1.0, progress)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    # ---- persist CATH vocab inside the Lightning checkpoint ------------- #
    def on_save_checkpoint(self, checkpoint):
        checkpoint["cath_vocab"] = self.cath_vocab.to_dict()


class PeriodicBenchmarkEvalCallback(Callback):
    def __init__(self, eval_steps: int = 2000):
        super().__init__()
        self.eval_steps = eval_steps
        self._last = -1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if self.eval_steps <= 0 or step == 0 or step % self.eval_steps != 0 or step == self._last:
            return
        self._last = step
        if trainer.strategy is not None:
            trainer.strategy.barrier("bench_start")
        if trainer.is_global_zero:
            metrics = pl_module.run_benchmark_evaluation()
            if metrics and trainer.logger is not None:
                table = metrics.pop("benchmark_samples_table", None)
                trainer.logger.log_metrics(metrics, step=step)
                if table is not None and isinstance(trainer.logger, WandbLogger):
                    trainer.logger.experiment.log({"benchmark_samples": table}, step=step)
                msg = (
                    f"[bench step {step}] ALL iou={metrics['val_subset_iou_chain']:.3f} "
                    f"correct={metrics['val_subset_correct_prop']:.3f} "
                    f"bds={metrics['val_subset_boundary_distance_score']:.3f} "
                    f"cath={metrics['val_subset_correct_cath']:.3f} "
                    f"cath_lvl={metrics['val_subset_cath_level_score']:.3f} "
                    f"parse_ok={metrics['val_subset_pred_parse_ok_rate']:.2f}"
                )
                if "val_subset_multi_iou_chain" in metrics:
                    msg += (
                        f" | MULTI(n={metrics.get('val_subset_multi_n', 0)}) "
                        f"iou={metrics['val_subset_multi_iou_chain']:.3f} "
                        f"correct={metrics['val_subset_multi_correct_prop']:.3f} "
                        f"cath={metrics['val_subset_multi_correct_cath']:.3f}"
                    )
                rank_zero_info(msg)
                del metrics, table
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        if trainer.strategy is not None:
            trainer.strategy.barrier("bench_end")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train two-stage TEDPred (ESM2 + seg + hierarchical CATH).")
    p.add_argument("--data_parquet_folder", type=str, default="data/all_parquet")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--max_len", type=int, default=1022)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--cath_min_count", type=int, default=5)
    # model
    p.add_argument("--esm_model_name", type=str, default="esm2_t33_650M_UR50D")
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num_seg_layers", type=int, default=4)
    p.add_argument("--dim_feedforward", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--pair_dim", type=int, default=128)
    p.add_argument("--cath_hidden", type=int, default=512)
    p.add_argument("--cath_cond_dim", type=int, default=64)
    p.add_argument("--esm_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--esm_chunk_size", type=int, default=1022)
    p.add_argument("--esm_chunk_overlap", type=int, default=256)
    # optim / recipe
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--residue_loss_weight", type=float, default=1.0)
    p.add_argument("--pair_loss_weight", type=float, default=1.0)
    p.add_argument("--cath_loss_weight", type=float, default=1.0)
    p.add_argument("--max_pair_len", type=int, default=768)
    # Tier 3: CATH long-tail handling (applied to the CATH head only)
    p.add_argument("--cath_loss_type", type=str, default="ce",
                   choices=["ce", "focal", "class_balanced", "cb_focal"],
                   help="ce=cross-entropy; focal; class_balanced=Cui et al. weights; "
                        "cb_focal=class-balanced focal")
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--cb_beta", type=float, default=0.999,
                   help="Class-balanced re-weighting beta (closer to 1 = stronger).")
    p.add_argument("--cath_level_weights", type=str, default="1,1,1,1",
                   help="Comma-separated static weights for C,A,T,H loss terms.")
    p.add_argument("--cath_curriculum_steps", type=int, default=0,
                   help="If >0, ramp deeper CATH levels in over this many steps (coarse-to-fine).")
    # clustering / eval
    p.add_argument("--cluster_method", type=str, default="connected_components",
                   choices=["connected_components", "spectral"])
    p.add_argument("--min_domain_len", type=int, default=20)
    p.add_argument("--benchmark_eval_steps", type=int, default=2000)
    p.add_argument("--benchmark_eval_subset_size", type=int, default=100)
    p.add_argument("--benchmark_eval_seed", type=int, default=42)
    p.add_argument("--benchmark_num_logged_samples", type=int, default=5)
    # io
    p.add_argument("--accelerator", type=str, default="auto")
    p.add_argument("--strategy", type=str, default="auto")
    p.add_argument("--save_dir", type=str, default="lightning_logs_two_stage")
    p.add_argument("--save_path", type=str, default="artifacts/two_stage_checkpoint.pt")
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    return p


def main():
    args = build_parser().parse_args()

    if args.accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("accelerator=gpu requested but CUDA is not available.")

    vocab_path = str(Path(args.save_dir) / "cath_vocab.json")
    dm = TwoStageDataModule(
        data_parquet_folder=args.data_parquet_folder,
        batch_size=args.batch_size,
        max_len=args.max_len,
        num_workers=args.num_workers,
        cath_min_count=args.cath_min_count,
        benchmark_eval_subset_size=args.benchmark_eval_subset_size,
        benchmark_eval_seed=args.benchmark_eval_seed,
        vocab_save_path=vocab_path,
    )
    dm.setup()

    level_weights = tuple(float(x) for x in str(args.cath_level_weights).split(","))
    # Class-balanced weights from TRAIN-set CATH frequencies (used only when the
    # loss type asks for them). Boundary losses stay on the natural distribution.
    cath_class_weights = None
    if args.cath_loss_type in ("class_balanced", "cb_focal"):
        cw = dm.cath_vocab.class_balanced_weights(beta=args.cb_beta)
        cath_class_weights = [torch.tensor(w, dtype=torch.float) for w in cw]
        rank_zero_info(
            "Class-balanced CATH weights enabled (beta="
            f"{args.cb_beta}); per-level weight ranges: "
            + ", ".join(f"L{l}[{min(w):.2f},{max(w):.2f}]" for l, w in enumerate(cw))
        )

    model = TwoStageLightningModule(
        cath_vocab=dm.cath_vocab,
        esm_model_name=args.esm_model_name,
        d_model=args.d_model, nhead=args.nhead, num_seg_layers=args.num_seg_layers,
        dim_feedforward=args.dim_feedforward, dropout=args.dropout, pair_dim=args.pair_dim,
        cath_hidden=args.cath_hidden, cath_cond_dim=args.cath_cond_dim,
        esm_dtype=args.esm_dtype, esm_chunk_size=args.esm_chunk_size,
        esm_chunk_overlap=args.esm_chunk_overlap,
        lr=args.lr, weight_decay=args.weight_decay, warmup_steps=args.warmup_steps,
        label_smoothing=args.label_smoothing,
        residue_loss_weight=args.residue_loss_weight,
        pair_loss_weight=args.pair_loss_weight,
        cath_loss_weight=args.cath_loss_weight,
        max_pair_len=args.max_pair_len,
        cluster_method=args.cluster_method, min_domain_len=args.min_domain_len,
        benchmark_num_logged_samples=args.benchmark_num_logged_samples,
        cath_loss_type=args.cath_loss_type,
        focal_gamma=args.focal_gamma,
        cath_level_weights=level_weights,
        cath_curriculum_steps=args.cath_curriculum_steps,
        cath_class_weights=cath_class_weights,
    )

    load_dotenv()
    wandb_key = os.getenv("WANDB_API_KEY")
    if wandb_key:
        wandb.login(key=wandb_key)
    wandb_logger = WandbLogger(
        project=os.getenv("WANDB_PROJECT", "ted-two-stage"),
        name=os.getenv("WANDB_RUN_NAME", "two-stage-esm2"),
        config=vars(args),
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=args.save_dir, filename="two_stage-{epoch:02d}-{eval_loss:.4f}",
        monitor="eval_loss", mode="min", save_top_k=1,
    )
    bench_cb = PeriodicBenchmarkEvalCallback(eval_steps=args.benchmark_eval_steps)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=1 if args.accelerator == "gpu" else "auto",
        strategy=args.strategy,
        logger=wandb_logger,
        callbacks=[ckpt_cb, bench_cb],
        gradient_clip_val=args.max_grad_norm,
        precision="bf16-mixed" if args.accelerator == "gpu" else 32,
    )

    trainer.fit(model, datamodule=dm, ckpt_path=args.resume_from_checkpoint)

    if args.save_path and trainer.is_global_zero:
        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.model.state_dict(),  # heads only (ESM excluded)
                "cath_vocab": dm.cath_vocab.to_dict(),
                "args": vars(args),
            },
            args.save_path,
        )
        print(f"Saved final checkpoint -> {args.save_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
