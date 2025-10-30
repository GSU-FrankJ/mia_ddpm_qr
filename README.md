# QR-MIA for Deterministic DDIM Membership Auditing

End-to-end, reproducible white-box membership inference pipeline targeting a DDIM trained from scratch on CIFAR-10. The workflow consists of deterministic data splits, DDIM training, t-error score extraction over uniform timesteps, quantile-regression threshold learning with bagging, and comprehensive evaluation with auditable reports.

## Environment Setup

```bash
# create a clean environment (PyTorch 2.1+ recommended)
conda create -n qr-mia python=3.10 -y
conda activate qr-mia
pip install -r requirements.txt

# install a CUDA build if desired, e.g. for CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

All scripts rely on PyTorch ≥ 2.1 and torchvision ≥ 0.16. The helper shell scripts assume execution from the repository root.

## One-Shot Pipeline

```bash
# 0) Deterministic train/aux/eval splits
bash scripts/split_cifar10.sh

# 1) Train DDIM (400k iterations by default; add --mode fastdev for a smoke run)
bash scripts/train_ddim_cifar10.sh --mode main --config configs/model_ddim.yaml --data configs/data_cifar10.yaml

# 2) Cache t-error scores for aux/eval sets (50 uniform steps by default)
bash scripts/compute_scores.sh --config configs/attack_qr.yaml

# 3) Train the quantile-regression ensemble with bagging (B=50, τ∈{1e-3,1e-4})
bash scripts/train_qr.sh --config configs/attack_qr.yaml

# 4) Evaluate and emit Markdown/JSON/plots under reports/<run_id>/
bash scripts/eval_qr.sh --config configs/attack_qr.yaml
```

Fast development mode is available on every Python entry point (`--fastdev`) to operate on ~1k images and five timesteps for quick sanity checks.

## Key Outputs

- `data/splits/*.json` — deterministic index lists (`member_train`, `eval_in`, `eval_out`, `aux`) guaranteeing zero leakage between training, auxiliary calibration, and evaluation.
- `runs/ddim_cifar10/<mode>/ckpt_*/` — DDIM checkpoints (`model.ckpt`, `ema.ckpt`, `optim.ckpt`) with `run.json` metadata (git hash, seeds, environment, AMP/determinism switches, TF32 posture).
- `scores/*.pt` — cached t-error tensors for auxiliary and evaluation sets with timestep metadata.
- `runs/attack_qr/ensembles/bagging.pt` — serialized bagging ensemble containing per-τ state dictionaries.
- `reports/<run_id>/` — Markdown + JSON summary, ROC figures, score/threshold histograms, and Parquet diagnostics (per-image evidence with score/log-score, thresholds per model, final decision, member label).

## Tests

```bash
python -m pytest
```

Tests cover the t-error shape/step selection, pinball loss monotonicity, bagging majority voting, split reproducibility, metric sanity, and dataset alignment utilities.

## Directory Overview

```
configs/              # YAML configs for data, model, and attack stages
data/                 # README plus reproducible split indices
ddpm_ddim/            # UNet model, cosine schedule, deterministic forward/reverse, trainer
attacks/              # t-error scoring, quantile datasets/models, bagging, evaluation tooling
scripts/              # Pipeline shells and Python CLIs
tests/                # Pytest suite ensuring determinism and metric correctness
reports/              # Generated during evaluation (ignored until created)
scores/               # Cached t-error tensors (generated after step 2)
```

## Notes

- The DDIM trainer defaults to EMA updates, AMP, and gradient clipping. Use `--mode fastdev` during early development to limit runtime.
- t-error computation is deterministic: no stochastic noise is injected (`η=0`), making the membership signal reproducible across hardware.
- Quantile regression uses pinball loss on `log1p` t-error scores with a short τ warm-up (default 3 epochs at τ=5e-3) before converging on ultra-low target quantiles; bagging mitigates variance and yields robust thresholds for very low false-positive regimes.
- Evaluation reports include bootstrap 95% confidence intervals for TPR and precision at FPR targets of 0.1% and 0.01%, plus ROC-AUC summaries.

## Ethical Use

This project is intended for privacy auditing and security research on publicly available datasets. Do not apply the codebase to sensitive or proprietary data without explicit authorization and adherence to local regulations.
