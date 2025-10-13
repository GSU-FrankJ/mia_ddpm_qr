# QR-MIA Diffusion Pipeline

End-to-end, deterministic white-box membership inference attack (MIA) for diffusion models using quantile regression with bagging (QR-MIA). The repository trains a DDPM on CIFAR-10, derives deterministic t-error features, fits quantile-regression weak learners, and evaluates white-box membership leakage.

## Features
- Deterministic dataset splits with saved indices for exact reproducibility.
- DDPM training (UNet, linear beta schedule) with EMA checkpoints and logging.
- Deterministic t-error computation (`eps` and `x0` modes) using Philox seeded noise.
- Quantile regression ensemble (ResNet-Tiny weak learners, bagging) predicting per-sample thresholds.
- White-box MIA evaluation with ROC/AUC, calibrated TPR@FPR (1% / 0.1%), bootstrap CIs, and artifact logging (CSV/JSON/Markdown).
- Unit tests covering pinball loss, deterministic t-errors, and split protocol.

## Setup
```bash
# Clone this repo, then create an environment
conda env create -f environment.yml
conda activate qr-mia
# or: pip install -r requirements.txt
```
> **Note:** PyTorch ≥ 2.1 and torchvision ≥ 0.16 are required. Install the CUDA build that matches your system, e.g.:
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

## Quickstart
All commands assume execution from the repository root.

```bash
# 0) Prepare splits (stores indices under runs/splits/)
python scripts/0_prepare_data.py --dataset cifar10 --seed 0 \
  --out runs/splits/cifar10_seed0.json

# 1) Train DDPM (creates runs/ddpm/cifar10/)
python scripts/1_train_ddpm.py --config configs/ddpm_cifar10.yaml \
  --split_json runs/splits/cifar10_seed0.json \
  --out runs/ddpm/cifar10

# 2) Build deterministic t-error pairs (NPZ + metadata)
python scripts/2_build_t_error_pairs.py --config configs/attack_qr.yaml \
  --split_json runs/splits/cifar10_seed0.json \
  --ddpm_ckpt runs/ddpm/cifar10/best.pt \
  --out runs/qr/pairs_cifar10.npz

# 3) Train quantile-regression ensemble
python scripts/3_train_qr_bagging.py --config configs/attack_qr.yaml \
  --pairs runs/qr/pairs_cifar10.npz \
  --out runs/qr/models

# 4) Evaluate white-box MIA
python scripts/4_eval_attack.py --config configs/eval.yaml \
  --split_json runs/splits/cifar10_seed0.json \
  --ddpm_ckpt runs/ddpm/cifar10/best.pt \
  --qr_dir runs/qr/models \
  --report runs/eval/cifar10_report.json
```
Expected ranges (dependent on training quality): `AUC ≳ 0.95`, `TPR@FPR=1% ≳ 0.7`, `TPR@FPR=0.1% ≳ 0.4` when using the provided configuration and a well-trained DDPM.

Artifacts:
- `runs/ddpm/<dataset>/`: checkpoints (`epoch_*.pt`, `best.pt`), training logs, metadata.
- `runs/qr/pairs_*.npz`: deterministic t-error pairs + metadata JSON.
- `runs/qr/models/`: ensemble checkpoints, manifest JSON.
- `runs/eval/`: raw scores (`csv/json`), summary table, final report.

## Reproducibility
- All runs use explicit seeds (see configs) and set deterministic CuDNN flags.
- Split JSON stores exact Z/Public/Holdout indices (`runs/splits/*`).
- Philox noise seeds are derived from `(dataset, sample_index, timestep, global_seed)` ensuring deterministic t-errors across devices.
- Bagging manifest logs per-model seeds and sampled bootstrap indices.

## Tests & Validation
```bash
python -m pytest
```
The current CI run exercises the deterministic t-error logic, pinball loss, and split protocol. Scripts were smoke-tested; ensure PyTorch/torchvision are installed before running data preparation (import failure indicates incompatible versions).

## Directory Layout
```
repo_root/
  configs/                # YAML configs for training/attack/eval
  ddpm/                   # Diffusion model code (models, schedules, engines)
  attack_qr/              # QR-MIA modules (features, models, engines, utils)
  scripts/                # CLI entrypoints 0-4 (pipeline stages)
  tests/                  # Unit tests
  runs/                   # Output directories (empty placeholder tracked)
  requirements.txt
  environment.yml
  README.md
  LICENSE
```

## Usage Notes & Tips
- Training duration: DDPM ~200 epochs on CIFAR-10 (expect several GPU hours). Adjust epochs/batch-size in `configs/ddpm_cifar10.yaml` for faster prototyping.
- To explore `x0` t-error mode, set `public.mode: x0` and pass `--mode x0` at evaluation.
- For additional datasets (e.g., CIFAR-100), replicate configs with a new dataset name; the split script handles stratification automatically.

## Ethics & Intended Use
This project is for privacy auditing and red-team research on generative models trained on consensual, public datasets. **Do not** apply the pipeline to sensitive or non-consensual data. Always follow local regulations and institutional review policies when evaluating privacy risks.
