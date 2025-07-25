import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from src.model.model import TransformerModel

class LitEdgeCompletion(pl.LightningModule):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.model = TransformerModel(vocab_size, cfg)
        self.cfg = cfg

    def forward(self, input_ids):
        return self.model(input_ids)

    def _step(self, batch, stage):
        logits = self(batch["input_ids"])
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch["labels"].view(-1), ignore_index=self.cfg.dataset.ignore_index)
        self.log(f"{stage}_loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.training.lr, weight_decay=self.cfg.training.weight_decay)