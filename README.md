# PPRD: Patch-Based Continual Learning with Contrastive Distillation

This repository provides a Split CIFAR-10 continual learning pipeline based on patch-level representations, replay, and PRD distillation.

Core features:

- Tensor-only 2x2 patch extraction and bilinear upsampling
- ResNet18 encoder + MLP projection head
- ISSupConLoss for contrastive supervision
- PRD logit distillation from the frozen previous model
- Patch prototype modes: class_mean_ema, class_confidence_ema, class_position_ema
- Unified backbone modes: patch and roi_patch
- Structured artifacts for downstream analysis, including stage diagnostics

## 1) Repository Layout

- train.py: main entrypoint for continual training and eval-only checkpoint analysis
- data/: split manager and replay buffer logic
- models/: patch/roi_patch backbone implementation
- utils/: losses, logging, and evaluation diagnostics utilities
- lab/: comparison runners and analysis notebook
- tests/: unit tests for diagnostics and replay behavior
- logs/: experiment outputs (generated)

## 2) Environment Setup

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

## 3) Dev Tooling

Install development tools:

```bash
pip install -r requirements-dev.txt
```

Run checks:

```bash
ruff check train.py data models utils lab tests
ruff format --check train.py data models utils lab tests
pytest
```

Enable pre-commit hooks:

```bash
pre-commit install
```

## 4) Quick Start

Smoke test:

```bash
python train.py --epochs 1 --linear-epochs 1 --batch-size 64 --num-workers 0 --max-train-batches 1 --max-eval-batches 1 --device auto --enable-csv
```

Backbone comparison:

```bash
python lab/run_backbone_comparison.py --backbones patch roi_patch --epochs 2 --linear-epochs 2 --batch-size 128 --replay-size 1000 --device auto
```

Patch prototype comparison:

```bash
python lab/run_patch_prototype_comparison.py --backbone patch --epochs 3 --linear-epochs 3
```

## 5) Device Backends

- --device auto: CUDA > MPS > CPU
- --device cuda: strict CUDA check
- --device mps: strict Apple Metal check
- --device cpu: force CPU

## 6) Run Artifacts

Each run directory contains:

- training.log: human-readable training timeline
- events.jsonl: structured events from LitLogger
- metrics.csv: scalar metrics from LitLogger
- results_tasks.json: per-stage linear-eval metrics over seen tasks
- results_step_eval.json: step-level raw-logit evaluation snapshots
- results_summary.json: final aggregate summary
- results_diagnostics.json: stage-level confusion diagnostics and failure flags
- results.json: combined artifact for compatibility

Important note:

- results_step_eval.json and results_tasks.json are different measurement modes.
- Step-eval uses raw model logits during training.
- Stage summary uses linear-probe evaluation on frozen features.

## 7) Notebook Analysis

Use lab/backbone_comparison.ipynb to:

- compare final per-task performance
- inspect all-task behavior across training stages (heatmap + trajectories)
- deep-dive near-chance failures using confusion diagnostics

If diagnostics are missing in existing runs, rerun the training/comparison scripts after updating train.py.

## 8) CI

GitHub Actions workflow is defined in .github/workflows/ci.yml and runs:

- Ruff lint/format checks
- Pytest suite
