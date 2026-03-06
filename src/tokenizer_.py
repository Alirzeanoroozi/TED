
# ---------------------------------------------------------------------------
# Tokenizer: build vocab from data (amino acids + chopping_star symbols)
# ---------------------------------------------------------------------------

import torch

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"


class TextTokenizer:
    """Character/token-level tokenizer for sequence and chopping_star."""

    def __init__(self, special_tokens=None):
        self.special_tokens = special_tokens or [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        self.token2id = {t: i for i, t in enumerate(self.special_tokens)}
        self.id2token = {i: t for t, i in self.token2id.items()}
        self.pad_id = self.token2id[PAD_TOKEN]
        self.sos_id = self.token2id[SOS_TOKEN]
        self.eos_id = self.token2id[EOS_TOKEN]
        self.unk_id = self.token2id[UNK_TOKEN]

    def fit(self, sequences):
        """Build vocabulary from list of strings."""
        for seq in sequences:
            for c in seq:
                if c not in self.token2id:
                    idx = len(self.token2id)
                    self.token2id[c] = idx
                    self.id2token[idx] = c
        return self

    def encode(self, text, add_sos_eos=False):
        ids = [self.token2id.get(c, self.unk_id) for c in text]
        if add_sos_eos:
            ids = [self.sos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids, strip_special=True):
        tokens = []
        for i in ids:
            if isinstance(i, torch.Tensor):
                i = i.item()
            if strip_special and i in (self.pad_id, self.sos_id, self.eos_id):
                continue
            tokens.append(self.id2token.get(i, UNK_TOKEN))
        return "".join(tokens)

    @property
    def vocab_size(self):
        return len(self.token2id)


