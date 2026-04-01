# Reproducibility Notes

## Determinism

The training script sets deterministic behavior where possible:

- Python `random` seed
- PyTorch seed
- PyTorch Lightning deterministic mode

Command-line seed:

```bash
python train.py --seed 7
```

## Hardware Selection

Use explicit backend to lock hardware:

- macOS GPU: `--device mps`
- Windows/Linux GPU: `--device cuda`
- CPU baseline: `--device cpu`

## Run Artifacts

Every run writes:

- `args.json` (full config)
- `run_command.txt`
- `run.sh`
- `model.txt`
- `results.json`
- `training.log`

These are sufficient for paper appendix reproducibility tables.
