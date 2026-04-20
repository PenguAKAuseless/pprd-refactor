# Patch Prototype Study

## Problem Statement

In PPRD, a prototype typically represents one class center. In patch-based processing, a class-level center can be too coarse because different patches in the same image can encode different semantics (object part, background, context).

This document defines candidate patch-prototype formulations, maps each to an implementation mode, and explains expected behavior.

## Notation

- Dataset samples: $(x_i, y_i)$
- Patch encoder output: $z_{i,n} \in \mathbb{R}^d$ for patch index $n \in \{1, \dots, N\}$
- Class label: $y_i \in \{1, \dots, C\}$
- Prototype bank at step $t$: $p^{(t)}$
- EMA momentum: $m \in [0,1)$

All prototype vectors are L2-normalized before use by PPRD.

## Candidate Definitions

### 1) Class Mean EMA (`class_mean_ema`)

Single prototype per class:

$$
\tilde{p}_c = \frac{1}{|\mathcal{S}_c|}\sum_{(i,n)\in\mathcal{S}_c} z_{i,n}, \quad
p_c^{(t)} = \operatorname{norm}(m\,p_c^{(t-1)} + (1-m)\,\operatorname{norm}(\tilde{p}_c))
$$

where $\mathcal{S}_c = \{(i,n) : y_i=c\}$.

- Pros: stable, simple, lowest variance.
- Cons: conflates part-specific structure into one center.

### 2) Confidence-Weighted Class EMA (`class_confidence_ema`)

Still one prototype per class, but patch contributions are weighted by class confidence.

Let $\ell_{i,n}$ be patch logits and

$$
w_{i,n} = \operatorname{softmax}(\alpha\,\ell_{i,n})_{y_i}
$$

Then

$$
\tilde{p}_c = \frac{\sum_{(i,n)\in\mathcal{S}_c} w_{i,n} z_{i,n}}{\sum_{(i,n)\in\mathcal{S}_c} w_{i,n}}
$$

EMA update is identical to mode (1). Here $\alpha$ is `patch_proto_sharpness`.

- Pros: suppresses noisy/background patches when classifier is calibrated.
- Cons: can reinforce early bias when confidence is overconfident but wrong.

### 3) Position-Aware EMA (`class_position_ema`)

Prototype per class and patch position: $p_{c,n}$.

$$
\tilde{p}_{c,n} = \frac{1}{|\mathcal{S}_{c,n}|}\sum_{i: y_i=c} z_{i,n}, \quad
p_{c,n}^{(t)} = \operatorname{norm}(m\,p_{c,n}^{(t-1)} + (1-m)\,\operatorname{norm}(\tilde{p}_{c,n}))
$$

with relation distillation over flattened bank
$\{p_{c,n}\}_{c=1..C, n=1..N}$.

- Pros: captures patch geometry/part specialization and increases prototype expressivity.
- Cons: larger relation space ($C\times N$ prototypes) can be noisier and harder to optimize in low-data regimes.

## Implementation Mapping

Implemented in a shared backbone pipeline:

- `models/patch_backbone.py` (`PrototypePatchBackbone`, mode=`patch|roi_patch`)
- `models/roi_patch_backbone.py` (compatibility wrapper around the shared implementation)

New arguments:

- `--patch-prototype-mode {class_mean_ema,class_confidence_ema,class_position_ema}`
- `--patch-proto-sharpness <float>` (used in `class_confidence_ema`)

Training integration:

- Patch indices are built as repeating `0..N-1` per image.
- `update_codebook` now receives patch logits and indices.
- `get_active_prototypes` returns:
  - shape `[C, d]` for class-level modes
  - shape `[C*N, d]` for position-aware mode

## How To Compare

Use:

```bash
python lab/run_patch_prototype_comparison.py \
  --backbone patch \
  --epochs 3 --linear-epochs 3 \
  --batch-size 128 --replay-size 1000
```

Fast smoke run:

```bash
python lab/run_patch_prototype_comparison.py \
  --backbone patch \
  --epochs 1 --linear-epochs 1 \
  --batch-size 64 --replay-size 200 \
  --max-train-batches 1 --max-eval-batches 1 --max-step-eval-batches 1 --step-eval-every 0
```

Output:

- `lab/results/patch_prototype_comparison.json`
- `lab/results/by_mode/*.json`

## Selection Guidance

Default recommendation for general use: start with `class_mean_ema`.

If underfitting fine-grained local structure: try `class_position_ema`.

If many patches are obvious clutter/background and classifier confidence is reasonably calibrated: try `class_confidence_ema` with `--patch-proto-sharpness` in $[0.8, 2.0]$.

## Why Math-Sound May Not Win Empirically

A mathematically richer definition can underperform due to optimization and data realities:

- Higher prototype count increases variance and distillation entropy.
- Replay imbalance can make some patch positions sparse.
- Confidence weighting depends on calibration quality, which is weak early in continual stages.
- EMA lag plus task shifts may delay adaptation of complex prototype banks.
