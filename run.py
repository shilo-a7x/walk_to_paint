import os
import pytorch_lightning as pl
from omegaconf import OmegaConf
import torch
from src.training.train import train_model
from src.data.prepare_data import prepare_all

def main():
    cfg = OmegaConf.load("config.yaml")
    print(OmegaConf.to_yaml(cfg))

    data_module, tokenizer = prepare_all(cfg)
    model = train_model(cfg, data_module)

if __name__ == "__main__":
    main()