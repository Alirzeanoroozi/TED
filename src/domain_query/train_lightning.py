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

from benchmark.ted_eval import EvalConfig, evaluate_target
from domain_query.config_utils import load_config
from domain_query.dataset import build_datasets, collate_domain_batch
from domain_query.decode import decode_batch
from domain_query.model import DomainQueryModel
from residue_mlm.labels import LevelVocabulary

torch.set_float32_matmul_precision("high")

CHAIN_METRICS = ("iou_chain", "correct_cath", "cath_level_score", "boundary_distance_score", "domain_count_match")

class DomainQueryDataModule(pl.LightningDataModule):
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
        self.cath_vocabs = None

    def setup(self, stage: str | None = None):
        if self.cath_vocabs is not None:
            return
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

    def _loader(self, dataset, shuffle: bool):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=collate_domain_batch,
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

class DomainQueryLightningModule(pl.LightningModule):
    def __init__(
        self,
        cath_vocab_sizes: dict[str, int],
        esm_model_name: str | None = "esm2_t33_650M_UR50D",
        freeze_esm: bool = True,
        max_len: int = 1022,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_queries: int = 16,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        loss_weights: dict[str, float] | None = None,
        decode_kwargs: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["cath_vocab_sizes", "loss_weights", "decode_kwargs"])
        self.loss_weights = loss_weights
        self.decode_kwargs = decode_kwargs or {}
        self.cath_vocabs: dict[str, LevelVocabulary] | None = None

        self.model = DomainQueryModel(
            cath_vocab_sizes=cath_vocab_sizes,
            esm_model_name=esm_model_name,
            freeze_esm=freeze_esm,
            max_len=max_len,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_queries=num_queries,
        )

    def set_cath_vocabs(self, cath_vocabs: dict[str, LevelVocabulary]) -> None:
        self.cath_vocabs = cath_vocabs

    def _eval_chain_metrics(self, outputs: dict, batch: dict) -> dict[str, float]:
        if self.cath_vocabs is None:
            return {}
        pred_chopping_stars, _ = decode_batch(
            outputs, batch["seq_len"], batch["left_offsets"], self.cath_vocabs, **self.decode_kwargs
        )
        cfg = EvalConfig(input_indexing="one_based_inclusive")
        totals = {key: 0.0 for key in CHAIN_METRICS}
        for pred, gt, seq_len in zip(pred_chopping_stars, batch["chopping_stars"], batch["seq_len"].tolist()):
            metrics = evaluate_target(gt, pred, nres=int(seq_len), config=cfg)
            for key in CHAIN_METRICS:
                totals[key] += float(metrics.get(key, 0.0))
        return {key: value / len(pred_chopping_stars) for key, value in totals.items()}

    def _shared_step(self, batch, stage: str):
        attention_mask = torch.arange(batch["aa_ids"].size(1), device=self.device).unsqueeze(0) < batch[
            "seq_len"
        ].unsqueeze(1)
        outputs = self.model(batch["sequences"], batch["aa_ids"], attention_mask)
        loss, metrics = self.model.compute_loss(
            outputs,
            batch["domain_masks"],
            batch["domain_cath"],
            batch["seq_len"],
            loss_weights=self.loss_weights,
        )
        if stage in ("val", "test"):
            metrics.update(self._eval_chain_metrics(outputs, batch))

        batch_size = batch["seq_len"].size(0)
        for key, value in metrics.items():
            log_key = f"{stage}_loss" if key == "loss" else f"{stage}_{key}"
            self.log(log_key, value, prog_bar=key in ("loss", "iou_chain"), batch_size=batch_size, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        params = [param for param in self.parameters() if param.requires_grad]
        return torch.optim.AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

def main():
    config_path = "configs/domain_query_config.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    args = load_config(config_path)

    dm = DomainQueryDataModule(
        data_parquet_folder=args.data_parquet_folder,
        batch_size=args.batch_size,
        max_len=args.max_len,
        val_size=args.val_size,
        test_size=args.test_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        random_crop_train=args.random_crop_train,
    )
    dm.setup()

    cath_vocab_sizes = {name: vocab.vocab_size for name, vocab in dm.cath_vocabs.items()}
    model = DomainQueryLightningModule(
        cath_vocab_sizes=cath_vocab_sizes,
        esm_model_name=args.esm_model_name,
        freeze_esm=args.freeze_esm,
        max_len=args.max_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        num_queries=args.num_queries,
        lr=args.lr,
        weight_decay=args.weight_decay,
        loss_weights={"mask": args.loss_weight_mask, "score": args.loss_weight_score, "cath": args.loss_weight_cath},
        decode_kwargs={
            "score_threshold": args.score_threshold,
            "max_gap": args.max_gap,
            "min_segment_len": args.min_segment_len,
        },
    )
    model.set_cath_vocabs(dm.cath_vocabs)

    wandb_logger = WandbLogger(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
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
