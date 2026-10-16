"""PyTorch Lightning trainer for residue-level MLM."""

from __future__ import annotations

import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from residue_mlm.config_utils import load_config
from residue_mlm.dataset import apply_mlm_mask, build_datasets, collate_residue_batch
from residue_mlm.labels import AA_MASK_ID, MASK_ID, SPECIAL_VOCAB_SIZE, AA_ALPHABET, LevelVocabulary
from residue_mlm.model import ResidueMLM
from residue_mlm.predict import (
    compute_residue_boundary_accuracy,
    compute_token_accuracies,
    evaluate_sequence_predictions,
    predict_residue_labels,
)

torch.set_float32_matmul_precision("high")


class ResidueMLMDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_parquet_folder: str,
        batch_size: int,
        max_len: int,
        val_size: int,
        test_size: int,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        random_crop_train: bool,
        mask_prob: float,
        mask_aa: bool,
        mask_cath: bool,
        mask_domain_id: bool,
    ):
        super().__init__()
        self.data_parquet_folder = data_parquet_folder
        self.batch_size = batch_size
        self.max_len = max_len
        self.val_size = val_size
        self.test_size = test_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.random_crop_train = random_crop_train
        self.mask_prob = mask_prob
        self.mask_aa = mask_aa
        self.mask_cath = mask_cath
        self.mask_domain_id = mask_domain_id
        self.cath_vocabs = None

    def setup(self, stage: str | None = None):
        train_ds, val_ds, test_ds, cath_vocabs = build_datasets(
            self.data_parquet_folder,
            max_len=self.max_len,
            val_size=self.val_size,
            test_size=self.test_size,
            random_crop_train=self.random_crop_train,
        )
        self.train_dataset = train_ds
        self.val_dataset = val_ds
        self.test_dataset = test_ds
        self.cath_vocabs = cath_vocabs

    def _collate(self, batch):
        return collate_residue_batch(batch, max_len=self.max_len)

    def _loader(self, dataset, shuffle: bool):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self._collate,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, shuffle=False)


class ResidueMLMLightningModule(pl.LightningModule):
    def __init__(
        self,
        cath_vocab_sizes: dict[str, int],
        max_len: int = 128,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_num_domains: int = 32,
        max_domain_index: int = 32,
        mask_prob: float = 0.15,
        mask_aa: bool = True,
        mask_cath: bool = True,
        mask_domain_id: bool = True,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        loss_weights: dict[str, float] | None = None,
        eval_sequence_prediction: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["cath_vocab_sizes", "loss_weights"])
        self.mask_prob = mask_prob
        self.mask_aa = mask_aa
        self.mask_cath = mask_cath
        self.mask_domain_id = mask_domain_id
        self.loss_weights = loss_weights or {}
        self.eval_sequence_prediction = eval_sequence_prediction
        self.cath_vocabs: dict[str, LevelVocabulary] | None = None

        self.model = ResidueMLM(
            max_len=max_len,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            aa_vocab_size=SPECIAL_VOCAB_SIZE + len(AA_ALPHABET),
            max_num_domains=max_num_domains,
            max_domain_index=max_domain_index,
            cath_vocab_sizes=cath_vocab_sizes,
        )

    def set_cath_vocabs(self, cath_vocabs: dict[str, LevelVocabulary]) -> None:
        self.cath_vocabs = cath_vocabs

    def _prepare_batch(self, batch: dict) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        level_mask_ids = {level: MASK_ID for level in ("cath_c", "cath_a", "cath_t", "cath_h")}

        tensor_batch = {key: value for key, value in batch.items() if torch.is_tensor(value)}
        masked = apply_mlm_mask(
            tensor_batch,
            mask_prob=self.mask_prob,
            aa_mask_id=AA_MASK_ID,
            level_mask_ids=level_mask_ids,
            mask_aa=self.mask_aa,
            mask_cath=self.mask_cath,
            mask_domain_id=self.mask_domain_id,
        )

        max_len = tensor_batch["positions"].size(1)
        attention_mask = torch.arange(max_len, device=tensor_batch["positions"].device).unsqueeze(0) < tensor_batch[
            "seq_len"
        ].unsqueeze(1)

        cath_inputs = {
            "cath_c": masked["cath_c_input"],
            "cath_a": masked["cath_a_input"],
            "cath_t": masked["cath_t_input"],
            "cath_h": masked["cath_h_input"],
        }

        model_inputs = {
            "aa_ids_input": masked["aa_ids_input"],
            "positions": tensor_batch["positions"],
            "num_domains": tensor_batch["num_domains"],
            "domain_ids_input": masked["domain_ids_input"],
            "cath_inputs": cath_inputs,
            "attention_mask": attention_mask,
        }

        labels = {
            "aa_labels": masked["aa_labels"],
            "domain_labels": masked["domain_labels"],
            "cath_c_labels": masked["cath_c_labels"],
            "cath_a_labels": masked["cath_a_labels"],
            "cath_t_labels": masked["cath_t_labels"],
            "cath_h_labels": masked["cath_h_labels"],
        }
        return model_inputs, labels, masked.get("mlm_mask")

    def _log_metrics(self, stage: str, metrics: dict[str, float], batch_size: int) -> None:
        for key, value in metrics.items():
            if key == "loss":
                log_key = "val_loss" if stage == "val" else f"{stage}_loss"
                self.log(log_key, value, prog_bar=True, batch_size=batch_size, sync_dist=True, on_epoch=True)
            else:
                self.log(
                    f"{stage}_{key}",
                    value,
                    prog_bar=key
                    in (
                        "domain_acc",
                        "cath_full_acc",
                        "residue_domain_acc",
                        "boundary_acc",
                        "iou_chain",
                        "correct_cath",
                        "cath_level_score",
                    ),
                    batch_size=batch_size,
                    sync_dist=True,
                    on_epoch=True,
                )

    def _eval_sequence_predictions(self, batch: dict, stage: str) -> dict[str, float]:
        if not self.eval_sequence_prediction or self.cath_vocabs is None:
            return {}

        tensor_batch = {key: value for key, value in batch.items() if torch.is_tensor(value)}
        predictions = predict_residue_labels(self.model, tensor_batch)
        boundary_metrics = compute_residue_boundary_accuracy(
            predictions["domain_ids"],
            tensor_batch["domain_ids"],
            tensor_batch["seq_len"],
        )
        chain_metrics = evaluate_sequence_predictions(
            self.model,
            tensor_batch,
            self.cath_vocabs,
            batch["chopping_stars"],
            batch["left_offsets"],
        )
        return {**boundary_metrics, **chain_metrics}

    def _shared_step(self, batch, stage: str):
        model_inputs, labels, mlm_mask = self._prepare_batch(batch)
        logits = self.model(**model_inputs)
        loss, metrics = self.model.compute_loss(logits, labels, loss_weights=self.loss_weights)

        acc_metrics = compute_token_accuracies(
            logits,
            labels,
            batch["seq_len"],
            mlm_mask=mlm_mask,
        )
        metrics.update(acc_metrics)

        if stage in ("val", "test"):
            metrics.update(self._eval_sequence_predictions(batch, stage))

        batch_size = batch["seq_len"].size(0)
        self._log_metrics(stage, metrics, batch_size)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)


def main():
    config_path = "configs/residue_mlm_config.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    args = load_config(config_path)

    dm = ResidueMLMDataModule(
        data_parquet_folder=args.data_parquet_folder,
        batch_size=args.batch_size,
        max_len=args.max_len,
        val_size=args.val_size,
        test_size=args.test_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        random_crop_train=args.random_crop_train,
        mask_prob=args.mask_prob,
        mask_aa=args.mask_aa,
        mask_cath=args.mask_cath,
        mask_domain_id=args.mask_domain_id,
    )
    dm.setup()

    cath_vocab_sizes = {name: vocab.vocab_size for name, vocab in dm.cath_vocabs.items()}
    loss_weights = {
        "aa": args.loss_weight_aa,
        "domain": args.loss_weight_domain,
        "cath_c": args.loss_weight_cath_c,
        "cath_a": args.loss_weight_cath_a,
        "cath_t": args.loss_weight_cath_t,
        "cath_h": args.loss_weight_cath_h,
    }

    model = ResidueMLMLightningModule(
        cath_vocab_sizes=cath_vocab_sizes,
        max_len=args.max_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_num_domains=args.max_num_domains,
        max_domain_index=args.max_domain_index,
        mask_prob=args.mask_prob,
        mask_aa=args.mask_aa,
        mask_cath=args.mask_cath,
        mask_domain_id=args.mask_domain_id,
        lr=args.lr,
        weight_decay=args.weight_decay,
        loss_weights=loss_weights,
        eval_sequence_prediction=args.eval_sequence_prediction,
    )
    model.set_cath_vocabs(dm.cath_vocabs)

    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_dir,
        filename=args.checkpoint_filename,
        monitor=args.checkpoint_monitor,
        mode=args.checkpoint_mode,
        save_top_k=args.save_top_k,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        gradient_clip_val=args.max_grad_norm,
    )
    trainer.fit(model, datamodule=dm)

    torch.save(
        {
            "model_state_dict": model.model.state_dict(),
            "cath_vocab_sizes": cath_vocab_sizes,
            "cath_vocabs": {name: vocab.token2id for name, vocab in dm.cath_vocabs.items()},
            "args": vars(args),
        },
        args.save_path,
    )
    print(f"Saved checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
