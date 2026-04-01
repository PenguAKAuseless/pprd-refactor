$ErrorActionPreference = 'Stop'

python train.py `
  --epochs 1 `
  --linear-epochs 1 `
  --batch-size 64 `
  --num-workers 0 `
  --max-train-batches 1 `
  --max-eval-batches 1 `
  --device auto `
  --enable-csv
