import json
from collections import Counter


class Tokenizer:
    PAD_TOKEN = "<PAD>"
    MASK_TOKEN = "<MASK>"
    UNK_TOKEN = "<UNK>"

    def __init__(self):
        self.token2id = {}
        self.id2token = {}
        self.vocab_size = 0

    def fit(self, sequences):
        counter = Counter(token for seq in sequences for token in seq)
        vocab = [self.PAD_TOKEN, self.MASK_TOKEN, self.UNK_TOKEN] + sorted(counter)

        self.token2id = {tok: i for i, tok in enumerate(vocab)}
        self.id2token = {i: tok for tok, i in self.token2id.items()}
        self.vocab_size = len(self.token2id)

    def encode(self, sequence):
        return [self.token2id.get(token, self.unk_idx) for token in sequence]

    def decode(self, ids):
        return [self.id2token.get(i, self.UNK_TOKEN) for i in ids]

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"token2id": self.token2id}, f)

    def load(self, path):
        with open(path) as f:
            self.token2id = json.load(f)["token2id"]
        self.id2token = {int(i): t for t, i in self.token2id.items()}
        self.vocab_size = len(self.token2id)

    @property
    def pad_idx(self):
        return self.token2id[self.PAD_TOKEN]

    @property
    def mask_idx(self):
        return self.token2id[self.MASK_TOKEN]

    @property
    def unk_idx(self):
        return self.token2id[self.UNK_TOKEN]
