import json


class Tokenizer:
    PAD = "<PAD>"
    MASK = "<MASK>"
    UNK = "<UNK>"
    UNK_LABEL = "<UNK_LABEL>"
    NODE_PREFIX = "N_"
    EDGE_PREFIX = "E_"
    DELIMITER = "_"

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
                self.add_token(f"{self.NODE_PREFIX}{u}")
                self.add_token(f"{self.NODE_PREFIX}{v}")
                self.add_token(f"{self.EDGE_PREFIX}{label}")
        self._build_edge_label_map()

    def _build_edge_label_map(self):
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
        return self.edge_label2id.get(token, self.UNK_LABEL_ID)

    def decode_edge_label(self, class_id):
        """Convert class ID (0, ..., num_classes-1) back to edge token string"""
        return self.id2edge_label.get(class_id, self.UNK_LABEL)

    def encode(self, sequence):
        if isinstance(sequence, str):
            sequence = [sequence]
        return [self.token2id.get(token, self.UNK_ID) for token in sequence]

    def decode(self, ids):
        if isinstance(ids, int):
            ids = [ids]
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

    def _token_or_id_to_str(self, token_or_id) -> str:
        if isinstance(token_or_id, str):
            return token_or_id
        return self.id2token.get(token_or_id, "")

    def is_edge(self, token_or_id):
        tok = self._token_or_id_to_str(token_or_id)
        return tok.startswith(self.EDGE_PREFIX)

    def is_node(self, token_or_id):
        tok = self._token_or_id_to_str(token_or_id)
        return tok.startswith(self.NODE_PREFIX)

    def parse_node(self, token_or_id) -> int:
        tok = self._token_or_id_to_str(token_or_id)
        return (
            int(tok.split(self.DELIMITER, 1)[1])
            if tok.startswith(self.NODE_PREFIX)
            else None
        )

    def parse_edge_label(self, token_or_id) -> int:
        tok = self._token_or_id_to_str(token_or_id)
        return (
            int(tok.split(self.DELIMITER, 1)[1])
            if tok.startswith(self.EDGE_PREFIX)
            else None
        )

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
    def UNK_LABEL_ID(self):
        return -1

    @property
    def vocab_size(self):
        return len(self.token2id)

    @property
    def num_edge_tokens(self):
        return len(self.edge_label2id)
