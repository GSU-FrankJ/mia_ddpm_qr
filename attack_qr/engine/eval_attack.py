from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from attack_qr.features.t_error import compute_t_error
from attack_qr.models.qr_resnet_tiny import ResNetTiny
from attack_qr.utils.metrics import bootstrap_metrics, compute_roc, interpolate_tpr
from attack_qr.utils.seeding import seed_everything, timesteps_seed
from ddpm.data.loader import IndexedDataset, get_dataset, get_transforms
from ddpm.schedules.noise import DiffusionSchedule


@dataclass
class EvalConfig:
    alpha: float
    mode: str = "eps"
    K: int = 4
    batch_size: int = 128
    bootstrap: int = 200
    seed: int = 0


def load_quantile_ensemble(models_dir: str | Path, device: torch.device) -> tuple[list[ResNetTiny], List[float]]:
    models_dir = Path(models_dir)
    manifest_path = models_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    alpha_list = manifest["alpha_list"]
    ensemble = []
    for entry in manifest["models"]:
        ckpt_path = models_dir / entry["path"]
        ckpt = torch.load(ckpt_path, map_location=device)
        model = ResNetTiny(num_outputs=len(alpha_list)).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        ensemble.append(model)
    return ensemble, alpha_list


def prepare_eval_dataloaders(
    dataset_name: str,
    data_root: str,
    member_indices: Sequence[int],
    nonmember_indices: Sequence[int],
    img_size: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    base = get_dataset(dataset_name, root=data_root, download=True)
    transform = get_transforms(dataset_name, img_size, augment=False)
    member_dataset = IndexedDataset(base, indices=member_indices, transform=transform)
    nonmember_dataset = IndexedDataset(base, indices=nonmember_indices, transform=transform)

    def _loader(ds):
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return _loader(member_dataset), _loader(nonmember_dataset)


def _average_t_error(
    ddpm_model: torch.nn.Module,
    schedule: DiffusionSchedule,
    images: torch.Tensor,
    indices: Sequence[int],
    dataset_name: str,
    global_seed: int,
    K: int,
    mode: str,
) -> torch.Tensor:
    device = images.device
    values = []
    for img, idx in zip(images, indices):
        idx_int = int(idx)
        rng = np.random.default_rng(timesteps_seed(dataset_name, idx_int, global_seed))
        timesteps = rng.integers(low=0, high=schedule.T, size=K, endpoint=False)
        x_batch = img.unsqueeze(0).repeat(K, 1, 1, 1)
        t_tensor = torch.as_tensor(timesteps, device=device, dtype=torch.long)
        err = compute_t_error(
            model=ddpm_model,
            schedule=schedule,
            x0=x_batch,
            timesteps=t_tensor,
            dataset_name=dataset_name,
            sample_indices=[idx_int] * K,
            global_seed=global_seed,
            mode=mode,
        )
        values.append(err.mean().item())
    return torch.tensor(values, device=device)


def evaluate_attack(
    ddpm_model: torch.nn.Module,
    schedule: DiffusionSchedule,
    ensemble: list[ResNetTiny],
    alpha_list: Sequence[float],
    config: EvalConfig,
    dataset_name: str,
    data_root: str,
    member_indices: Sequence[int],
    nonmember_indices: Sequence[int],
    img_size: int,
    global_seed: int,
    device: str | torch.device = "cuda",
    out_dir: str | Path = "runs/eval",
) -> Dict:
    device = torch.device(device)
    seed_everything(config.seed)
    ddpm_model.to(device).eval()
    schedule = schedule.to(device)
    for model in ensemble:
        model.to(device).eval()

    try:
        alpha_idx = alpha_list.index(config.alpha)
    except ValueError as exc:
        raise ValueError(f"Alpha {config.alpha} not in ensemble outputs {alpha_list}") from exc

    members_loader, nonmembers_loader = prepare_eval_dataloaders(
        dataset_name=dataset_name,
        data_root=data_root,
        member_indices=member_indices,
        nonmember_indices=nonmember_indices,
        img_size=img_size,
        batch_size=config.batch_size,
    )

    records = []

    def _process(loader, label):
        for images, _, idxs in tqdm(loader, desc=f"Eval label={label}"):
            images = images.to(device, non_blocking=True)
            l_bar = _average_t_error(ddpm_model, schedule, images, idxs, dataset_name, global_seed, config.K, config.mode)
            preds = [model(images).detach()[:, alpha_idx] for model in ensemble]
            preds_tensor = torch.stack(preds, dim=0)
            margins = preds_tensor - l_bar.unsqueeze(0)
            vote_counts = (margins > 0).sum(dim=0)
            score = margins.mean(dim=0)
            for idx_val, s_val, votes, t_err in zip(idxs.tolist(), score.cpu().tolist(), vote_counts.cpu().tolist(), l_bar.cpu().tolist()):
                records.append(
                    {
                        "image_id": int(idx_val),
                        "label": int(label),
                        "score": float(s_val),
                        "vote_count": int(votes),
                        "t_error": float(t_err),
                    }
                )

    _process(members_loader, label=1)
    _process(nonmembers_loader, label=0)

    labels = np.array([r["label"] for r in records], dtype=np.int32)
    scores = np.array([r["score"] for r in records], dtype=np.float32)

    roc = compute_roc(labels, scores)
    target_fprs = [0.01, 0.001]
    interpolated = {f: float(interpolate_tpr(roc.fprs, roc.tprs, f)) for f in target_fprs}

    negatives = scores[labels == 0]
    positives = scores[labels == 1]

    def threshold_at_fpr(target: float) -> tuple[float, float, float]:
        if target <= 0:
            return float("inf"), 0.0, 0.0
        sorted_scores = np.sort(negatives)[::-1]
        k = max(1, int(math.ceil(target * len(sorted_scores))))
        threshold = sorted_scores[k - 1]
        actual_fpr = float((negatives > threshold).mean())
        tpr = float((positives > threshold).mean())
        return threshold, actual_fpr, tpr

    calibrated = {}
    for fpr in target_fprs:
        thr, act_fpr, tpr = threshold_at_fpr(fpr)
        calibrated[fpr] = {
            "threshold": float(thr),
            "fpr": float(act_fpr),
            "tpr": float(tpr),
        }

    boot = bootstrap_metrics(labels, scores, target_fprs=target_fprs, n_bootstrap=config.bootstrap, seed=config.seed)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_scores.json"
    with raw_path.open("w", encoding="utf-8") as f:
        json.dump(records, f)
    with (out_dir / "raw_scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "label", "score", "vote_count", "t_error"])
        writer.writeheader()
        for row in records:
            writer.writerow(row)

    report = {
        "dataset": dataset_name,
        "alpha": config.alpha,
        "mode": config.mode,
        "K": config.K,
        "M": len(ensemble),
        "metrics": {
            "auc": float(roc.auc),
            "tpr_at": interpolated,
            "calibrated": calibrated,
            "bootstrap": boot,
        },
    }

    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    summary_row = {
        "dataset": dataset_name,
        "alpha": config.alpha,
        "mode": config.mode,
        "M": len(ensemble),
        "K": config.K,
        "AUC": roc.auc,
        "TPR@1%": interpolated[0.01],
        "TPR@0.1%": interpolated[0.001],
    }
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)
    md = (
        "| dataset | alpha | mode | M | K | AUC | TPR@1% | TPR@0.1% |\n"
        "|---|---|---|---|---|---|---|---|\n"
        f"| {dataset_name} | {config.alpha} | {config.mode} | {len(ensemble)} | {config.K} | {roc.auc:.4f} | "
        f"{interpolated[0.01]:.4f} | {interpolated[0.001]:.4f} |\n"
    )
    (out_dir / "summary.md").write_text(md, encoding="utf-8")

    return report
