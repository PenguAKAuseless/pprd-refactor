# PPRD: Patch-Based Prototype Relation Distillation for Continual Learning

This repository implements a research-grade Split CIFAR-10 continual learning pipeline built on patch-level representations, replay, and prototype relation distillation. The codebase is organized into modular building blocks (encoders, patch extractors, heads, and codebooks) and produces standardized artifacts for reproducible analysis.

## Highlights

- Modular, lego-block model assembly via `models/builder.py`
- Patch and ROI patch extractors with shared code paths
- ETF classifier and configurable prototype codebooks (EMA or fixed)
- Replay buffer and importance-scaled supervised contrastive loss
- Standardized output artifacts for task-eval and step-eval analysis
- WandB sweep support for loss balancing

## Repository Layout

- `train.py`: main entrypoint for continual training and eval-only analysis
- `data/`: split manager and replay buffer logic
- `models/`: encoders, extractors, heads, and model builder
- `utils/`: losses, logging, and evaluation diagnostics
- `lab/`: ablation runners and analysis notebooks
- `logs/`: experiment outputs (generated)

## Environment Setup

Option A (pip):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Option B (conda):

```bash
conda env create -f environment.yml
conda activate pprd
```

Dev tooling:

```bash
pip install -r requirements-dev.txt
ruff check train.py data models utils lab
ruff format --check train.py data models utils lab
pytest
```

## Quick Start

Smoke test:

```bash
python train.py --epochs 1 --linear-epochs 1 --batch-size 64 --num-workers 0 \
  --max-train-batches 1 --max-eval-batches 1 --device cpu
```

Standard continual run (normal patches + EMA mean codebook):

```bash
python train.py --patch-mode normal --codebook-mode ema_mean \
  --epochs 10 --batch-size 128 --replay-size 500
```

ROI patches with confidence-weighted prototypes:

```bash
python train.py --patch-mode roi --patch-prototype-mode class_confidence_ema \
  --epochs 10 --batch-size 128 --replay-size 500
```

## Ablation Script

`run_ablation.sh` executes the three target configurations with unique run directories. You can override defaults via environment variables:

```bash
EPOCHS=2 LINEAR_EPOCHS=2 BATCH_SIZE=128 REPLAY_SIZE=500 DEVICE=auto ./run_ablation.sh
```

## Analysis Notebook

Use `lab/analysis.ipynb` for post-hoc analysis only. It includes `fetch_latest_experiment()` to locate the most recent run matching a configuration and utilities for Average Accuracy (AA) and Backward Transfer (BWT).

Important separation:

- Task-eval (linear probe after each stage) lives in `results_tasks.json`.
- Step-eval (raw logits during training) lives in `results_step_eval.json`.

## Run Artifacts

Each run directory contains:

- `training.log`: training timeline
- `events.jsonl`: structured events from LitLogger
- `metrics.csv`: scalar metric time series
- `results_tasks.json`: per-stage task-eval metrics plus a uniform `task_eval` row
- `results_step_eval.json`: step-level raw-logit snapshots
- `results_summary.json`: final aggregate metrics (includes `final_avg_accuracy`)
- `results_diagnostics.json`: confusion diagnostics and failure flags
- `results.json`: combined artifact for compatibility
- `model_task_{N}.pth`, `model_final.pth`: checkpoints
- `args.json`: CLI arguments used for the run

## WandB Sweeps

A Bayesian sweep configuration is provided in `sweep.yaml`. Example usage:

```bash
wandb sweep sweep.yaml
wandb agent <entity>/<project>/<sweep_id>
```

The sweep metric is logged as `sweep/avg_accuracy` (Average Accuracy over all tasks at the final stage).

## Device Backends

- `--device auto`: CUDA > MPS > CPU
- `--device cuda`: strict CUDA check
- `--device mps`: strict Apple Metal check
- `--device cpu`: force CPU
