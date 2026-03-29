import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from dotenv import load_dotenv
import wandb

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from data.dataset_ import collate_fn, create_train_val_test_datasets, _load_paths
from tokenizer_ import TextTokenizer
from model import TextToTextTransformer
from evaluate import greedy_decode

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from benchmark.ted_eval import EvalConfig, evaluate_target
except ImportError:
    from ted_eval import EvalConfig, evaluate_target


def resolve_parquet_inputs(data_parquet_path: str) -> list[str]:
    path = Path(data_parquet_path).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    else:
        path = path.resolve()

    if path.is_file():
        if path.suffix.lower() != ".parquet":
            raise ValueError(f"Expected a parquet file, got: {path}")
        parquet_files = [str(path)]
    elif path.is_dir():
        parquet_files = sorted(str(parquet_file) for parquet_file in path.glob("*.parquet"))
    else:
        raise FileNotFoundError(f"Data path does not exist: {path}")

    if not parquet_files:
        raise RuntimeError(f"No .parquet files found under {path}")

    return parquet_files


class TEDDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_parquet_folder: str,
        batch_size: int = 8,
        max_src_len: int = 512,
        max_tgt_len: int = 256,
        num_workers: int = 0,
        benchmark_eval_subset_size: int = 100,
        benchmark_eval_fixed_subset: bool = True,
        benchmark_eval_seed: int = 42,
    ):
        super().__init__()
        self.data_parquet_folder = data_parquet_folder
        self.batch_size = batch_size
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.num_workers = num_workers
        self.benchmark_eval_subset_size = benchmark_eval_subset_size
        self.benchmark_eval_fixed_subset = benchmark_eval_fixed_subset
        self.benchmark_eval_seed = benchmark_eval_seed
        self.src_tokenizer = None
        self.tgt_tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.benchmark_val_subset = None
        self._benchmark_subset_calls = 0

    def setup(self, stage: str | None = None):
        parquet_files = resolve_parquet_inputs(self.data_parquet_folder)
        # Fit tokenizers on a subset of data (same columns as dataset: sequence, chopping_star)
        df = _load_paths(parquet_files[:3] if len(parquet_files) >= 3 else parquet_files)
        self.src_tokenizer = TextTokenizer().fit(df["sequence"].astype(str).tolist())
        self.tgt_tokenizer = TextTokenizer().fit(df["chopping_star"].astype(str).tolist())
        self.train_dataset, self.val_dataset, self.test_dataset = create_train_val_test_datasets(
            parquet_files,
            self.src_tokenizer,
            self.tgt_tokenizer,
            self.max_src_len,
            self.max_tgt_len,
        )

        if self.val_dataset is not None and len(self.val_dataset) > 0:
            self.benchmark_val_subset = Subset(self.val_dataset, self._select_benchmark_indices())

    def _select_benchmark_indices(self):
        subset_size = min(self.benchmark_eval_subset_size, len(self.val_dataset))
        if subset_size <= 0:
            return []

        seed = self.benchmark_eval_seed
        if not self.benchmark_eval_fixed_subset:
            seed += self._benchmark_subset_calls
            self._benchmark_subset_calls += 1

        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(self.val_dataset), generator=generator)[:subset_size].tolist()
        return indices

    def _build_loader(self, dataset, shuffle: bool):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=lambda b: collate_fn(b, self.src_tokenizer.pad_id, self.tgt_tokenizer.pad_id),
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self._build_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._build_loader(self.val_dataset, shuffle=False)

    def benchmark_val_dataloader(self):
        if self.val_dataset is None or len(self.val_dataset) == 0:
            return None
        if not self.benchmark_eval_fixed_subset:
            self.benchmark_val_subset = Subset(self.val_dataset, self._select_benchmark_indices())
        return self._build_loader(self.benchmark_val_subset, shuffle=False)


class TEDLightningModule(pl.LightningModule):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        tgt_pad_id: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_src_len: int = 512,
        max_tgt_len: int = 256,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        benchmark_num_logged_samples: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["tgt_pad_id"])
        self.tgt_pad_id = tgt_pad_id
        self.model = TextToTextTransformer(
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
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_id)
        self.benchmark_eval_config = EvalConfig(input_indexing="one_based_inclusive")

    def forward(self, src, tgt_in, src_key_padding_mask=None, tgt_key_padding_mask=None):
        return self.model(
            src,
            tgt_in,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

    def _move_batch_to_device(self, batch):
        moved = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def _shared_step(self, batch, src_pad_id, tgt_pad_id):
        src = batch["src"]
        tgt_in = batch["tgt_in"]
        tgt_out = batch["tgt_out"]
        src_key_padding_mask = src == src_pad_id
        tgt_key_padding_mask = tgt_in == tgt_pad_id
        logits = self(src, tgt_in, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        return loss

    def training_step(self, batch, batch_idx):
        src_pad_id = self.trainer.datamodule.src_tokenizer.pad_id
        tgt_pad_id = self.trainer.datamodule.tgt_tokenizer.pad_id
        loss = self._shared_step(batch, src_pad_id, tgt_pad_id)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src_pad_id = self.trainer.datamodule.src_tokenizer.pad_id
        tgt_pad_id = self.trainer.datamodule.tgt_tokenizer.pad_id
        loss = self._shared_step(batch, src_pad_id, tgt_pad_id)
        self.log("eval_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def _mean_metric(self, values):
        if not values:
            return float("nan")
        return float(np.nanmean(values))

    def _build_benchmark_samples_table(self, sample_rows):
        if not sample_rows:
            return None

        table = wandb.Table(
            columns=[
                "input_sequence",
                "target_chopping_star",
                "target_chopping_star_model",
                "target_truncated_for_model",
                "predicted_chopping_star",
                "gt_parse_ok",
                "pred_parse_ok",
                "gt_parse_errors",
                "pred_parse_errors",
                "gt_parse_warnings",
                "pred_parse_warnings",
                "gt_domain_count",
                "pred_domain_count",
                "domain_count_match",
                "iou_chain",
                "ndo",
                "correct_prop",
                "correct_cath",
                "boundary_distance_score",
            ]
        )
        for sample in sample_rows:
            table.add_data(
                sample["input_sequence"],
                sample["target_chopping_star"],
                sample["target_chopping_star_model"],
                sample["target_truncated_for_model"],
                sample["predicted_chopping_star"],
                sample["gt_parse_ok"],
                sample["pred_parse_ok"],
                sample["gt_parse_errors"],
                sample["pred_parse_errors"],
                sample["gt_parse_warnings"],
                sample["pred_parse_warnings"],
                sample["gt_domain_count"],
                sample["pred_domain_count"],
                sample["domain_count_match"],
                sample["iou_chain"],
                sample["ndo"],
                sample["correct_prop"],
                sample["correct_cath"],
                sample["boundary_distance_score"],
            )
        return table

    def run_benchmark_evaluation(self, dataloader=None):
        datamodule = self.trainer.datamodule
        dataloader = dataloader or datamodule.benchmark_val_dataloader()
        if dataloader is None:
            return None

        src_pad_id = datamodule.src_tokenizer.pad_id
        tgt_pad_id = datamodule.tgt_tokenizer.pad_id
        tgt_tokenizer = datamodule.tgt_tokenizer
        was_training = self.training
        metric_values = {
            "iou_chain": [],
            "ndo": [],
            "correct_prop": [],
            "correct_cath": [],
            "boundary_distance_score": [],
        }
        loss_sum = 0.0
        item_count = 0
        sample_rows = []
        max_logged_samples = max(0, int(self.hparams.benchmark_num_logged_samples))
        gt_parse_ok_count = 0
        pred_parse_ok_count = 0
        domain_count_match_count = 0
        gt_truncated_for_model_count = 0

        self.eval()
        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_batch_to_device(batch)
                batch_size = batch["src"].size(0)
                loss = self._shared_step(batch, src_pad_id, tgt_pad_id)
                loss_sum += float(loss.detach().cpu()) * batch_size
                item_count += batch_size

                generated_ids = greedy_decode(
                    self.model,
                    batch["src"],
                    src_pad_id=src_pad_id,
                    tgt_pad_id=tgt_pad_id,
                    sos_id=tgt_tokenizer.sos_id,
                    eos_id=tgt_tokenizer.eos_id,
                    max_len=self.hparams.max_tgt_len,
                    device=self.device,
                )

                predictions = [
                    tgt_tokenizer.decode(row.tolist(), strip_special=True)
                    for row in generated_ids.detach().cpu()
                ]

                for src_text, tgt_text, tgt_text_model, tgt_was_truncated, pred_text in zip(
                    batch["src_text"],
                    batch["tgt_text"],
                    batch["tgt_text_model"],
                    batch["tgt_was_truncated"],
                    predictions,
                ):
                    metrics = evaluate_target(
                        tgt_text,
                        pred_text,
                        nres=len(src_text),
                        sequence=src_text,
                        config=self.benchmark_eval_config,
                    )
                    metric_values["iou_chain"].append(float(metrics["iou_chain"]))
                    metric_values["ndo"].append(float(metrics["ndo"]))
                    metric_values["correct_prop"].append(float(metrics["correct_prop"]))
                    metric_values["correct_cath"].append(float(metrics["correct_cath"]))
                    metric_values["boundary_distance_score"].append(float(metrics["boundary_distance_score"]))
                    gt_parse_ok_count += int(bool(metrics["gt_parse_ok"]))
                    pred_parse_ok_count += int(bool(metrics["pred_parse_ok"]))
                    domain_count_match_count += int(bool(metrics["domain_count_match"]))
                    gt_truncated_for_model_count += int(bool(tgt_was_truncated))

                    if len(sample_rows) < max_logged_samples:
                        sample_rows.append(
                            {
                                "input_sequence": src_text,
                                "target_chopping_star": tgt_text,
                                "target_chopping_star_model": tgt_text_model,
                                "target_truncated_for_model": bool(tgt_was_truncated),
                                "predicted_chopping_star": pred_text,
                                "gt_parse_ok": bool(metrics["gt_parse_ok"]),
                                "pred_parse_ok": bool(metrics["pred_parse_ok"]),
                                "gt_parse_errors": metrics["gt_parse_errors"],
                                "pred_parse_errors": metrics["pred_parse_errors"],
                                "gt_parse_warnings": metrics["gt_parse_warnings"],
                                "pred_parse_warnings": metrics["pred_parse_warnings"],
                                "gt_domain_count": int(metrics["gt_domain_count"]),
                                "pred_domain_count": int(metrics["pred_domain_count"]),
                                "domain_count_match": bool(metrics["domain_count_match"]),
                                "iou_chain": float(metrics["iou_chain"]),
                                "ndo": float(metrics["ndo"]),
                                "correct_prop": float(metrics["correct_prop"]),
                                "correct_cath": float(metrics["correct_cath"]),
                                "boundary_distance_score": float(metrics["boundary_distance_score"]),
                            }
                        )

        if was_training:
            self.train()

        if item_count == 0:
            return None

        return {
            "eval_loss": loss_sum / item_count,
            "val_subset_eval_loss": loss_sum / item_count,
            "val_subset_iou_chain": self._mean_metric(metric_values["iou_chain"]),
            "val_subset_ndo": self._mean_metric(metric_values["ndo"]),
            "val_subset_correct_prop": self._mean_metric(metric_values["correct_prop"]),
            "val_subset_correct_cath": self._mean_metric(metric_values["correct_cath"]),
            "val_subset_boundary_distance_score": self._mean_metric(metric_values["boundary_distance_score"]),
            "val_subset_gt_parse_ok_rate": gt_parse_ok_count / item_count,
            "val_subset_pred_parse_ok_rate": pred_parse_ok_count / item_count,
            "val_subset_domain_count_match_rate": domain_count_match_count / item_count,
            "val_subset_target_truncated_for_model_rate": gt_truncated_for_model_count / item_count,
            "benchmark_samples_table": self._build_benchmark_samples_table(sample_rows),
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer


class PeriodicBenchmarkEvalCallback(Callback):
    def __init__(self, eval_steps: int = 1000):
        super().__init__()
        self.eval_steps = eval_steps
        self._last_eval_step = -1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        global_step = trainer.global_step
        if self.eval_steps <= 0 or global_step == 0:
            return
        if global_step % self.eval_steps != 0 or global_step == self._last_eval_step:
            return

        self._last_eval_step = global_step

        if trainer.strategy is not None:
            trainer.strategy.barrier("benchmark_eval_start")

        if trainer.is_global_zero:
            metrics = pl_module.run_benchmark_evaluation()
            if metrics and trainer.logger is not None:
                sample_table = metrics.pop("benchmark_samples_table", None)
                trainer.logger.log_metrics(metrics, step=global_step)
                if sample_table is not None and isinstance(trainer.logger, WandbLogger):
                    trainer.logger.experiment.log({"benchmark_samples": sample_table}, step=global_step)
                rank_zero_info(
                    "Benchmark eval step "
                    f"{global_step}: eval_loss={metrics['eval_loss']:.4f}, "
                    f"iou_chain={metrics['val_subset_iou_chain']:.4f}, "
                    f"ndo={metrics['val_subset_ndo']:.4f}, "
                    f"correct_prop={metrics['val_subset_correct_prop']:.4f}, "
                    f"correct_cath={metrics['val_subset_correct_cath']:.4f}, "
                    f"boundary_distance_score={metrics['val_subset_boundary_distance_score']:.4f}, "
                    f"pred_parse_ok_rate={metrics['val_subset_pred_parse_ok_rate']:.4f}, "
                    f"target_truncated_for_model_rate={metrics['val_subset_target_truncated_for_model_rate']:.4f}"
                )

        if trainer.strategy is not None:
            trainer.strategy.barrier("benchmark_eval_end")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_parquet_folder",
        type=str,
        default="data/all_parquet",
        help="Path to a parquet directory or a single parquet file.",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Per-GPU batch size")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_encoder_layers", type=int, default=4)
    parser.add_argument("--num_decoder_layers", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_src_len", type=int, default=1024)
    parser.add_argument("--max_tgt_len", type=int, default=256)
    parser.add_argument("--save_path", type=str, default="transformer_checkpoint.pt", help="Final checkpoint path (PyTorch .pt)")
    parser.add_argument("--save_dir", type=str, default="lightning_logs", help="Lightning checkpoint dir")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--strategy", type=str, default="auto", help="e.g. ddp, ddp_spawn, auto")
    parser.add_argument("--benchmark_eval_steps", type=int, default=1000, help="Run benchmark evaluation every N global training steps")
    parser.add_argument("--benchmark_eval_subset_size", type=int, default=100, help="Number of validation examples used for periodic benchmark evaluation")
    parser.add_argument("--benchmark_eval_seed", type=int, default=42, help="Seed for selecting the validation benchmark subset")
    parser.add_argument(
        "--benchmark_num_logged_samples",
        type=int,
        default=5,
        help="Number of qualitative benchmark examples to log to W&B on each periodic evaluation",
    )
    parser.add_argument(
        "--benchmark_eval_random_subset",
        action="store_true",
        help="Resample the validation benchmark subset on each periodic evaluation instead of reusing a fixed subset",
    )
    args = parser.parse_args()

    if args.accelerator == "gpu":
        if torch.version.cuda is None:
            raise RuntimeError(
                "PyTorch in the active environment is a CPU-only build "
                f"(torch=={torch.__version__}, torch.version.cuda=None). "
                "Install a CUDA-enabled PyTorch build in the current environment and rerun."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "PyTorch was built with CUDA support, but no CUDA device is available to this job. "
                "Check the Slurm GPU allocation and CUDA runtime visibility."
            )

    dm = TEDDataModule(
        data_parquet_folder=args.data_parquet_folder,
        batch_size=args.batch_size,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        num_workers=args.num_workers,
        benchmark_eval_subset_size=args.benchmark_eval_subset_size,
        benchmark_eval_fixed_subset=not args.benchmark_eval_random_subset,
        benchmark_eval_seed=args.benchmark_eval_seed,
    )

    dm.setup()

    model = TEDLightningModule(
        src_vocab_size=dm.src_tokenizer.vocab_size,
        tgt_vocab_size=dm.tgt_tokenizer.vocab_size,
        tgt_pad_id=dm.tgt_tokenizer.pad_id,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        lr=args.lr,
        weight_decay=args.weight_decay,
        benchmark_num_logged_samples=args.benchmark_num_logged_samples,
    )

    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        raise RuntimeError(
            "WANDB_API_KEY is not set. Export it in the shell or source it from the Slurm job "
            "before running training."
        )

    wandb.login(key=wandb_api_key)
    wandb_logger = WandbLogger(
        project=os.getenv("WANDB_PROJECT", "ted-transformer"),
        name=os.getenv("WANDB_RUN_NAME", "ted-transformer-lightning"),
        config=vars(args),
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_dir,
        filename="ted-{epoch:02d}-{eval_loss:.4f}",
        monitor="eval_loss",
        mode="min",
        save_top_k=1,
    )
    benchmark_callback = PeriodicBenchmarkEvalCallback(eval_steps=args.benchmark_eval_steps)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=1 if args.accelerator == "gpu" else "auto",
        strategy=args.strategy,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, benchmark_callback],
        gradient_clip_val=args.max_grad_norm,
    )

    trainer.fit(model, datamodule=dm)

    if args.save_path and trainer.is_global_zero:
        torch.save(
            {
                "model_state_dict": model.model.state_dict(),
                "src_vocab_size": dm.src_tokenizer.vocab_size,
                "tgt_vocab_size": dm.tgt_tokenizer.vocab_size,
                "args": vars(args),
            },
            args.save_path,
        )
        print(f"Saved checkpoint to {args.save_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
