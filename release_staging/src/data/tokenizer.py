import json
import warnings
import numpy as np


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

        # Cache sets for fast lookups
        self._edge_tokens = set()
        self._node_tokens = set()
        self._token_to_node_id = {}
        self._token_to_edge_label = {}

    def add_token(self, token):
        if token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token

            # Update cached lookup structures
            if token.startswith(self.EDGE_PREFIX):
                self._edge_tokens.add(token)
                try:
                    label = int(token.split(self.DELIMITER, 1)[1])
                    self._token_to_edge_label[token] = label
                except (IndexError, ValueError):
                    warnings.warn(
                        f"Tokenizer: cannot parse integer label from edge token {token!r}; "
                        "it will not appear in _token_to_edge_label and will get sort key 0 "
                        "in _build_edge_label_map (may cause incorrect class assignment).",
                        stacklevel=2,
                    )
            elif token.startswith(self.NODE_PREFIX):
                self._node_tokens.add(token)
                try:
                    node_id = int(token.split(self.DELIMITER, 1)[1])
                    self._token_to_node_id[token] = node_id
                except (IndexError, ValueError):
                    warnings.warn(
                        f"Tokenizer: cannot parse integer node id from node token {token!r}; "
                        "it will not appear in _token_to_node_id.",
                        stacklevel=2,
                    )

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
        """Create mapping from edge label tokens to [0, num_classes).

        Tokens are sorted by their numeric sign value (ascending) so that the
        most negative sign always gets class 0 and the most positive sign always
        gets the highest class ID.  For binary datasets (-1/+1):
            class 0 = distrust (E_-1), class 1 = trust (E_1)  ← conventional
        Safe for multi-label: -10 → 0, -9 → 1, ..., +10 → 20.
        """
        self.edge_label2id = {}
        self.id2edge_label = {}
        edge_tokens = sorted(
            self._edge_tokens,
            key=lambda t: self._token_to_edge_label.get(t, 0),
        )
        for current, token in enumerate(edge_tokens):
            self.edge_label2id[token] = current
            self.id2edge_label[current] = token

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
        """Optimized encoding using direct dictionary access"""
        if isinstance(sequence, str):
            return [self.token2id.get(sequence, self.UNK_ID)]
        # Fast batch encoding
        unk_id = self.UNK_ID
        token2id = self.token2id
        return [token2id.get(token, unk_id) for token in sequence]

    def encode_batch(self, sequences):
        """Vectorized batch encoding for maximum performance"""
        unk_id = self.UNK_ID
        token2id = self.token2id
        return [[token2id.get(token, unk_id) for token in seq] for seq in sequences]

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

        # Rebuild cached lookup structures
        tok._edge_tokens = set()
        tok._node_tokens = set()
        tok._token_to_node_id = {}
        tok._token_to_edge_label = {}

        for token in tok.token2id.keys():
            if token.startswith(tok.EDGE_PREFIX):
                tok._edge_tokens.add(token)
                try:
                    label = int(token.split(tok.DELIMITER, 1)[1])
                    tok._token_to_edge_label[token] = label
                except (IndexError, ValueError):
                    pass
            elif token.startswith(tok.NODE_PREFIX):
                tok._node_tokens.add(token)
                try:
                    node_id = int(token.split(tok.DELIMITER, 1)[1])
                    tok._token_to_node_id[token] = node_id
                except (IndexError, ValueError):
                    pass

        return tok

    def _token_or_id_to_str(self, token_or_id) -> str:
        if isinstance(token_or_id, str):
            return token_or_id
        return self.id2token.get(token_or_id, "")

    def is_edge(self, token_or_id):
        """Optimized edge checking using cached set"""
        if isinstance(token_or_id, str):
            return token_or_id in self._edge_tokens
        tok = self.id2token.get(token_or_id, "")
        return tok in self._edge_tokens

    def is_node(self, token_or_id):
        """Optimized node checking using cached set"""
        if isinstance(token_or_id, str):
            return token_or_id in self._node_tokens
        tok = self.id2token.get(token_or_id, "")
        return tok in self._node_tokens

    def parse_node(self, token_or_id) -> int:
        """Optimized node parsing using cached mapping"""
        if isinstance(token_or_id, str):
            return self._token_to_node_id.get(token_or_id, None)
        tok = self.id2token.get(token_or_id, "")
        return self._token_to_node_id.get(tok, None)

    def parse_edge_label(self, token_or_id) -> int:
        """Optimized edge label parsing using cached mapping"""
        if isinstance(token_or_id, str):
            return self._token_to_edge_label.get(token_or_id, None)
        tok = self.id2token.get(token_or_id, "")
        return self._token_to_edge_label.get(tok, None)

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
