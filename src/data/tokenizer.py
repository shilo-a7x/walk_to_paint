import json


class Tokenizer:
    PAD = "<PAD>"
    MASK = "<MASK>"
    UNK = "<UNK>"

    def __init__(self):
        self.token2id = {
            self.PAD: 0,
            self.MASK: 1,
            self.UNK: 2,
        }
        self.id2token = {v: k for k, v in self.token2id.items()}

    def add_token(self, token):
        if token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token

    def fit(self, walks, edges=None):
        for walk in walks:
            for tok in walk:
                self.add_token(tok)
        if edges:
            for u, v, label in edges:
                self.add_token(f"N_{u}")
                self.add_token(f"N_{v}")
                self.add_token(f"E_{label}")
        self.build_edge_label_map()

    def build_edge_label_map(self):
        """Create mapping from edge label tokens to [0, num_classes)"""
        self.edge_label2id = {}
        self.id2edge_label = {}
        current = 0
        for token in self.token2id:
            if self.is_edge(token):
                self.edge_label2id[token] = current
                self.id2edge_label[current] = token
                current += 1

    def encode_edge_label(self, token_or_id):
        """Convert edge token (str or int) to class ID in [0, num_classes)"""
        token = (
            token_or_id
            if isinstance(token_or_id, str)
            else self.id2token.get(token_or_id, "")
        )
        return self.edge_label2id.get(token, -1)

    def decode_edge_label(self, class_id):
        """Convert class ID (0, ..., num_classes-1) back to edge token string"""
        return self.id2edge_label.get(class_id, self.UNK)

    def encode(self, sequence):
        return [self.token2id.get(token, self.UNK_ID) for token in sequence]

    def decode(self, ids):
        return [self.id2token.get(i, self.UNK) for i in ids]

    def save(self, path):
        with open(path, "w") as f:
            json.dump(
                {"token2id": self.token2id, "edge_label2id": self.edge_label2id}, f
            )

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        tok = cls()
        tok.token2id = data.get("token2id", {})
        tok.id2token = {int(v): k for k, v in tok.token2id.items()}
        tok.edge_label2id = data.get("edge_label2id", {})
        tok.id2edge_label = {v: k for k, v in tok.edge_label2id.items()}

        return tok

    def is_edge(self, token_or_id):
        if isinstance(token_or_id, int):
            token = self.id2token.get(token_or_id, "")
        else:
            token = token_or_id
        return token.startswith("E_")

    @property
    def PAD_ID(self):
        return self.token2id[self.PAD]

    @property
    def MASK_ID(self):
        return self.token2id[self.MASK]

    @property
    def UNK_ID(self):
        return self.token2id[self.UNK]

    @property
    def vocab_size(self):
        return len(self.token2id)

    @property
    def num_edge_tokens(self):
        return len([k for k in self.token2id if self.is_edge(k)])
