# Edge Masked Completion with Transformers

This project classifies edge types in graphs by treating random walks as masked sentences and training a Transformer to predict the edge labels.

## Features
- Random walk generation with masking
- Transformer encoder for sequence modeling
- PyTorch Lightning training pipeline
- OmegaConf-based config system
- Train/Val/Test support with evaluation
- Multiple dataset support
- TensorBoard logging

## Quick Start

```bash
pip install -r requirements.txt
python run.py --config config.yaml
```

To override parameters:

```bash
python run.py --config config.yaml training.lr=0.0005 dataset.name=bitcoin
```