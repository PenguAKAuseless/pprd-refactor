# PPRD: Patch-based Continual Learning with Contrastive Distillation

This repository contains a PyTorch + PyTorch Lightning implementation for continual learning on Split CIFAR-10 with:

- Tensor-only 2x2 patch extraction and bilinear upsampling
- ResNet18 backbone + MLP projection head
- ISSupConLoss for contrastive training
- PRD logits distillation from frozen old model
- Multi-backend runtime support: CUDA (Windows/Linux), MPS (macOS), CPU fallback
- Unified experiment logging: Lightning loggers + LitLogger + optional W&B

## 1) Repository Layout

- `train.py`: Main entrypoint for continual training and linear evaluation
- `models/`: Model definitions (`patch_backbone.py`)
- `data/`: Dataset and replay management (`datasets.py`)
- `utils/`: Losses and logging utilities (`losses.py`, `litlogger.py`)
- `configs/`: Reproducible command templates and baseline configurations
- `scripts/`: Cross-platform run helpers
- `docs/`: Experiment, logging, and reproducibility docs

## 2) Environment Setup

### Option A: pip (recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install --upgrade pip
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: Conda

```bash
conda env create -f environment.yml
conda activate pprd
```

## 3) W&B Setup

1. Copy `.env.example` to `.env`
2. Fill `WANDB_API_KEY`, `WANDB_ENTITY`, and optional project defaults
3. Run with `--use-wandb`

Example:

```bash
python train.py --use-wandb --wandb-project pprd --wandb-tags split-cifar10,baseline
```

## 4) Quick Start

Smoke test:

```bash
python train.py --epochs 1 --linear-epochs 1 --batch-size 64 --num-workers 0 --max-train-batches 1 --max-eval-batches 1 --device auto --enable-csv --enable-tb
```

Full run:

```bash
python train.py --epochs 5 --linear-epochs 5 --batch-size 128 --replay-size 1000 --device auto --enable-csv --enable-tb --use-wandb
```

## 5) Device Backends

- `--device auto`: CUDA > MPS > CPU
- `--device cuda`: strict CUDA check
- `--device mps`: strict Apple Metal check
- `--device cpu`: force CPU

Examples:

```bash
python train.py --device mps
python train.py --device cuda
```

## 6) Logging Outputs

Each run directory contains:

- `training.log`: human-readable run timeline
- `events.jsonl`: structured event log from LitLogger
- `metrics.csv`: task-level metrics from LitLogger
- `csv_logs/`: Lightning CSV logger outputs (if enabled)
- `tb_logs/`: TensorBoard logs (if enabled)
- `results.json`: per-task summary

See `docs/LOGGING.md` for details.
