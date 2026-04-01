# Logging and Evaluation Guide

## Logging Stack

This project supports layered logging for robust experiment tracking:

- Lightning loggers:
  - CSVLogger (`--enable-csv`)
  - TensorBoardLogger (`--enable-tb`)
  - WandbLogger (`--use-wandb`)
- LitLogger (`utils/litlogger.py`):
  - `events.jsonl` for structured events
  - `metrics.csv` for task-level metrics
- Plain text timeline:
  - `training.log`

## Recommended Flags

```bash
python train.py \
  --enable-csv \
  --enable-tb \
  --use-wandb \
  --wandb-project pprd \
  --wandb-tags split-cifar10,cl
```

## Evaluation Records

Per task, the training loop logs:

- `task/avg_train_loss`
- `task/linear_eval_acc`

Saved to:

- `results.json`
- W&B metrics (if enabled)
- LitLogger `metrics.csv`

## Checkpointing

Lightning saves task-level checkpoints under:

- `checkpoints/`

Current policy:

- `save_top_k=1`
- `save_last=True`
- monitor: `train/loss_epoch`
