import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from config_utills import load_config
from data._dataset import collate_fn, load_data, resolve_parquet_inputs, build_sample_weights, split_datasets
from data._tokenizer import TextTokenizer
from model.model import TextToTextTransformer
from evaluate_ import greedy_decode, grammar_guided_decode
from benchmark.ted_eval import EvalConfig, evaluate_target

torch.set_float32_matmul_precision("high")

class TEDDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_parquet_folder,
        batch_size,
        max_src_len,
        max_tgt_len,
        weighted_sampling,
        sampling_alpha,
        num_workers,
        pin_memory,
        persistent_workers,
    ):
        super().__init__()
        self.data_parquet_folder = data_parquet_folder
        self.batch_size = batch_size
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.weighted_sampling = weighted_sampling
        self.sampling_alpha = sampling_alpha
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.train_sample_weights = None
        
    def setup(self, stage: str | None = None):
        parquet_files = resolve_parquet_inputs(self.data_parquet_folder)    
        df = load_data(parquet_files[:3])
        self.src_tokenizer = TextTokenizer().fit(df["sequence"].astype(str).tolist())
        self.tgt_tokenizer = TextTokenizer().fit(df["chopping_star"].astype(str).tolist())
        self.train_dataset, self.val_dataset, self.test_dataset = split_datasets(parquet_files, self.src_tokenizer, self.tgt_tokenizer, self.max_src_len, self.max_tgt_len)

        if self.weighted_sampling:
            self.train_sample_weights = build_sample_weights(self.train_dataset.df, alpha=self.sampling_alpha)

    def _build_loader(self, dataset, shuffle: bool, sampler=None):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            collate_fn=lambda b: collate_fn(b, src_pad_id=self.src_tokenizer.pad_id, tgt_pad_id=self.tgt_tokenizer.pad_id),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
        )

    def train_dataloader(self):
        if self.weighted_sampling and self.train_sample_weights is not None:
            sampler = WeightedRandomSampler(weights=self.train_sample_weights, num_samples=len(self.train_sample_weights), replacement=True)
            return self._build_loader(self.train_dataset, shuffle=False, sampler=sampler)
        return self._build_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._build_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._build_loader(self.test_dataset, shuffle=False)

class TEDLightningModule(pl.LightningModule):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_pad_id: int,
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
        grammar_guided_decoding: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id
        self.grammar_guided_decoding = grammar_guided_decoding
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
        return self.model(src, tgt_in, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)

    def _move_batch_to_device(self, batch):
        moved = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def _shared_step(self, batch):
        src = batch["src"]
        tgt_in = batch["tgt_in"]
        tgt_out = batch["tgt_out"]
        src_key_padding_mask = src == self.src_pad_id
        tgt_key_padding_mask = tgt_in == self.tgt_pad_id
        logits = self(src, tgt_in, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch["src"].size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("eval_loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch["src"].size(0))
        return loss

    def _mean_metric(self, values):
        if not values:
            return float("nan")
        return float(np.nanmean(values))

    def _run_single_decode_pass(self, dataloader):
        tgt_tokenizer = self.trainer.datamodule.tgt_tokenizer

        metric_values = {"iou_chain": [], "ndo": [], "correct_prop": [], "correct_cath": [], "cath_level_score": [], "boundary_distance_score": []}
        loss_sum = 0.0
        item_count = 0
        sample_rows = []
        gt_parse_ok_count = 0
        pred_parse_ok_count = 0
        domain_count_match_count = 0

        for batch in dataloader:
            batch = self._move_batch_to_device(batch)
            batch_size = batch["src"].size(0)
            loss = self._shared_step(batch)
            loss_sum += float(loss.detach().cpu()) * batch_size
            item_count += batch_size

            generated_ids = grammar_guided_decode(
                self.model,
                batch["src"],
                src_pad_id=self.src_pad_id,
                tgt_pad_id=self.tgt_pad_id,
                sos_id=tgt_tokenizer.sos_id,
                eos_id=tgt_tokenizer.eos_id,
                max_len=self.hparams.max_tgt_len,
                device=self.device,
                token2id=tgt_tokenizer.token2id,
                id2token=tgt_tokenizer.id2token,
            )

            predictions = [tgt_tokenizer.decode(row.tolist(), strip_special=True) for row in generated_ids.detach().cpu()]

            for src_text, tgt_text, pred_text in zip(batch["src_text"], batch["tgt_text"], predictions):
                metrics = evaluate_target(tgt_text, pred_text, nres=len(src_text), sequence=src_text, config=self.benchmark_eval_config)
                metric_values["iou_chain"].append(float(metrics["iou_chain"]))
                metric_values["ndo"].append(float(metrics["ndo"]))
                metric_values["correct_prop"].append(float(metrics["correct_prop"]))
                metric_values["correct_cath"].append(float(metrics["correct_cath"]))
                metric_values["cath_level_score"].append(float(metrics["cath_level_score"]))
                metric_values["boundary_distance_score"].append(float(metrics["boundary_distance_score"]))
                gt_parse_ok_count += int(bool(metrics["gt_parse_ok"]))
                pred_parse_ok_count += int(bool(metrics["pred_parse_ok"]))
                domain_count_match_count += int(bool(metrics["domain_count_match"]))

                if len(sample_rows) < 5:
                    sample_rows.append({
                        "input_sequence": src_text,
                        "target_chopping_star": tgt_text,
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
                        "cath_level_score": float(metrics["cath_level_score"]),
                        "boundary_distance_score": float(metrics["boundary_distance_score"]),
                    })

        return {
            "metric_values": metric_values,
            "loss_sum": loss_sum,
            "item_count": item_count,
            "sample_rows": sample_rows,
            "gt_parse_ok_count": gt_parse_ok_count,
            "pred_parse_ok_count": pred_parse_ok_count,
            "domain_count_match_count": domain_count_match_count,
        }

    def _summarize_pass(self, raw: dict) -> dict:
        """Average the raw accumulated data from one decode pass."""
        n = raw["item_count"]
        mv = raw["metric_values"]
        return {
            "eval_loss": raw["loss_sum"] / n,
            "iou_chain": self._mean_metric(mv["iou_chain"]),
            "ndo": self._mean_metric(mv["ndo"]),
            "correct_prop": self._mean_metric(mv["correct_prop"]),
            "correct_cath": self._mean_metric(mv["correct_cath"]),
            "cath_level_score": self._mean_metric(mv["cath_level_score"]),
            "boundary_distance_score": self._mean_metric(mv["boundary_distance_score"]),
            "gt_parse_ok_rate": raw["gt_parse_ok_count"] / n,
            "pred_parse_ok_rate": raw["pred_parse_ok_count"] / n,
            "domain_count_match_rate": raw["domain_count_match_count"] / n,
            # "target_truncated_for_model_rate": raw["gt_truncated_for_model_count"] / n,
        }

    def _build_samples_table(self, greedy_rows, guided_rows=None):
        """Build a W&B table.

        When guided_rows is provided the table shows greedy and guided
        predictions side-by-side so the PI can compare them directly.
        """
        if not greedy_rows:
            return None

        base_cols = [
            "input_sequence", "target_chopping_star", "gt_parse_ok",
            "gt_domain_count", "gt_parse_errors",
        ]
        greedy_cols = [
            "greedy_predicted", "greedy_parse_ok", "greedy_domain_count",
            "greedy_iou_chain", "greedy_ndo", "greedy_correct_prop",
            "greedy_correct_cath", "greedy_cath_level_score", "greedy_boundary_distance_score",
        ]
        guided_cols = [
            "guided_predicted", "guided_parse_ok", "guided_domain_count",
            "guided_iou_chain", "guided_ndo", "guided_correct_prop",
            "guided_correct_cath", "guided_cath_level_score", "guided_boundary_distance_score",
        ] if guided_rows else []

        table = wandb.Table(columns=base_cols + greedy_cols + guided_cols)

        for i, g_row in enumerate(greedy_rows):
            base = [
                g_row["input_sequence"], g_row["target_chopping_star"],
                # g_row["target_chopping_star_model"], g_row["target_truncated_for_model"],
                g_row["gt_parse_ok"], g_row["gt_domain_count"], g_row["gt_parse_errors"],
            ]
            greedy = [
                g_row["predicted_chopping_star"], g_row["pred_parse_ok"],
                g_row["pred_domain_count"], g_row["iou_chain"], g_row["ndo"],
                g_row["correct_prop"], g_row["correct_cath"],
                g_row["cath_level_score"], g_row["boundary_distance_score"],
            ]
            if guided_rows and i < len(guided_rows):
                gu_row = guided_rows[i]
                guided = [
                    gu_row["predicted_chopping_star"], gu_row["pred_parse_ok"],
                    gu_row["pred_domain_count"], gu_row["iou_chain"], gu_row["ndo"],
                    gu_row["correct_prop"], gu_row["correct_cath"],
                    gu_row["cath_level_score"], gu_row["boundary_distance_score"],
                ]
            else:
                guided = []
            table.add_data(*(base + greedy + guided))

        return table

    def run_benchmark_evaluation(self):
        val_dataloader = self.trainer.datamodule.val_dataloader()
        self.eval()
        with torch.no_grad():
            metrics = self._run_single_decode_pass(val_dataloader)
        self.train()

        if metrics["item_count"] == 0:
            return None

        summary = self._summarize_pass(metrics)

        out = {
            "eval_loss": summary["eval_loss"],
            "val_subset_eval_loss": summary["eval_loss"],
            "val_subset_gt_parse_ok_rate": summary["gt_parse_ok_rate"],
            "val_subset_iou_chain": summary["iou_chain"],
            "val_subset_ndo": summary["ndo"],
            "val_subset_correct_prop": summary["correct_prop"],
            "val_subset_correct_cath": summary["correct_cath"],
            "val_subset_cath_level_score": summary["cath_level_score"],
            "val_subset_boundary_distance_score": summary["boundary_distance_score"],
            "val_subset_pred_parse_ok_rate": summary["pred_parse_ok_rate"],
            "val_subset_domain_count_match_rate": summary["domain_count_match_rate"],
        }

        out["benchmark_samples_table"] = self._build_samples_table(metrics["sample_rows"])

        return out

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

class PeriodicBenchmarkEvalCallback(Callback):
    def __init__(self, eval_steps: int = 1000):
        super().__init__()
        self.eval_steps = eval_steps
        self._last_eval_step = -1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        global_step = trainer.global_step
        if self.eval_steps <= 0 or global_step == 0 or global_step % self.eval_steps != 0 or global_step == self._last_eval_step:
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

def main():
    config = "configs/lightning_config.yaml"
    args = load_config(config)

    dm = TEDDataModule(
        data_parquet_folder=args.data_parquet_folder,
        batch_size=args.batch_size,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        weighted_sampling=args.weighted_sampling,
        sampling_alpha=args.sampling_alpha,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )
    dm.setup()

    model = TEDLightningModule(
        src_vocab_size=dm.src_tokenizer.vocab_size,
        tgt_vocab_size=dm.tgt_tokenizer.vocab_size,
        src_pad_id=dm.src_tokenizer.pad_id,
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
        grammar_guided_decoding=args.grammar_guided_decoding,
    )

    wandb_logger = WandbLogger(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_dir,
        filename=args.checkpoint_filename,
        monitor=args.checkpoint_monitor,
        mode=args.checkpoint_mode,
        save_top_k=args.save_top_k,
    )
    benchmark_callback = PeriodicBenchmarkEvalCallback(eval_steps=args.benchmark_eval_steps)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, benchmark_callback],
        gradient_clip_val=args.max_grad_norm,
    )
    trainer.fit(model, datamodule=dm)

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

if __name__ == "__main__":
    main()
