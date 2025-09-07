from torch.utils.data import DataLoader
from dataset import BilingualDataset

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from pathlib import Path
from tqdm import tqdm
import os
os.makedirs("../tokenizers", exist_ok=True)

train_path = "../dlp/jsons/ted_train.json"
validation_path = "../dlp/jsons/ted_validation.json"

def get_all_sentences(ds, src_tgt):
    for item in ds:
        yield item[src_tgt]

def get_or_build_tokenizer(tokenizer_path, ds, lang, vocab_size):
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        # tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], vocab_size=vocab_size)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    # It only has the train split, so we divide it overselves
    raw_ds = load_dataset(
        "json",
        data_files={
            "train": train_path,
            "validation": validation_path,
        },
    )

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(Path("../tokenizers/tokenizer_src.json"), raw_ds["train"], "Sequence", config['src_vocab_size'])
    tokenizer_tgt = get_or_build_tokenizer(Path("../tokenizers/tokenizer_tgt.json"), raw_ds["train"], "label", config['tgt_vocab_size'])
    
    train_ds = BilingualDataset(raw_ds["train"], tokenizer_src, tokenizer_tgt, "Sequence", "label", config['src_seq_len'])
    val_ds = BilingualDataset(raw_ds["validation"], tokenizer_src, tokenizer_tgt, "Sequence", "label", config['tgt_seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    # max_len_src = 0
    # max_len_tgt = 0

    # for item in tqdm(raw_ds["train"]):
    #     src_ids = tokenizer_src.encode(item["Sequence"]).ids
    #     tgt_ids = tokenizer_tgt.encode(item["label"]).ids
    #     max_len_src = max(max_len_src, len(src_ids))
    #     max_len_tgt = max(max_len_tgt, len(tgt_ids))

    # print(f'Max length of source sentence: {max_len_src}')
    # print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt