import os
import argparse
import torch
import torch.nn as nn
from dotenv import load_dotenv
import wandb

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from data.dataset_ import collate_fn, create_train_val_test_datasets, _load_paths
from tokenizer_ import TextTokenizer
from model import TextToTextTransformer

class TEDDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_parquet_folder: str,
        batch_size: int = 8,
        max_src_len: int = 512,
        max_tgt_len: int = 256,
        num_workers: int = 0,
    ):
        super().__init__()
        self.data_parquet_folder = data_parquet_folder
        self.batch_size = batch_size
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.num_workers = num_workers
        self.src_tokenizer = None
        self.tgt_tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str | None = None):
        # Read all Parquet files from the provided folder
        parquet_files = [
            os.path.join(self.data_parquet_folder, f)
            for f in os.listdir(self.data_parquet_folder)
            if f.endswith(".parquet")
        ]
        parquet_files = sorted(parquet_files)
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

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, self.src_tokenizer.pad_id, self.tgt_tokenizer.pad_id),
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, self.src_tokenizer.pad_id, self.tgt_tokenizer.pad_id),
            num_workers=self.num_workers,
            pin_memory=True,
        )

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

    def forward(self, src, tgt_in, src_key_padding_mask=None, tgt_key_padding_mask=None):
        return self.model(
            src,
            tgt_in,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

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
        # Get pad ids from the dataloader's dataset
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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_parquet_folder", type=str, default="data/parquet_sequences", help="Path to data.parquet")
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
    parser.add_argument("--max_tgt_len", type=int, default=128)
    parser.add_argument("--save_path", type=str, default="transformer_checkpoint.pt", help="Final checkpoint path (PyTorch .pt)")
    parser.add_argument("--save_dir", type=str, default="lightning_logs", help="Lightning checkpoint dir")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--strategy", type=str, default="auto", help="e.g. ddp, ddp_spawn, auto")
    args = parser.parse_args()

    dm = TEDDataModule(
        data_parquet_folder=args.data_parquet_folder,
        batch_size=args.batch_size,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        num_workers=args.num_workers,
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
    )

    load_dotenv()
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb_logger = WandbLogger(
        project="ted-transformer",
        name="ted-transformer-lightning",
        config=vars(args),
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_dir,
        filename="ted-{epoch:02d}-{eval_loss:.4f}",
        monitor="eval_loss",
        mode="min",
        save_top_k=1,
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=torch.cuda.device_count(),
        strategy=args.strategy,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        gradient_clip_val=args.max_grad_norm,
    )

    trainer.fit(model, datamodule=dm)

    # Save final PyTorch checkpoint
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
