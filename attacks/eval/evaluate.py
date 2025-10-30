"""Evaluate QR-MIA ensemble and generate report."""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Dict, List

import sys

import torch
from torch.utils.data import DataLoader, Subset
import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mia_logging import get_winston_logger
from attacks.qr.bagging import BagOfQuantiles
from attacks.scores.compute_scores import SplitDataset, load_indices
from attacks.eval.metrics import roc_auc, tpr_precision_at_fpr
from attacks.eval import report as report_module


LOGGER = get_winston_logger(__name__)


def load_yaml(path: pathlib.Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_loader(
    data_cfg: Dict,
    split_name: str,
    indices: List[int],
    limit: int | None = None,
) -> DataLoader:
    root = pathlib.Path(data_cfg["dataset"]["root"])
    mean = tuple(data_cfg["dataset"]["normalization"]["mean"])
    std = tuple(data_cfg["dataset"]["normalization"]["std"])
    train_flag = split_name != "aux"
    dataset = SplitDataset(root, indices, train=train_flag, mean=mean, std=std)
    if limit is not None and limit < len(dataset):
        dataset = Subset(dataset, list(range(limit)))
    return DataLoader(
        dataset,
        batch_size=data_cfg["training"].get("batch_size", 256),
        shuffle=False,
        num_workers=data_cfg["dataset"].get("num_workers", 8),
    )


def compute_detection_stats(
    ensemble: BagOfQuantiles,
    loader: DataLoader,
    scores: torch.Tensor,
    tau: float,
    member_label: int,
) -> Dict:
    margins = []
    decisions = []
    thresholds_log = []
    thresholds_raw = []
    votes = []
    scores_raw = []
    scores_log = []
    cursor = 0
    for images, _ in loader:
        batch_size = images.size(0)
        batch_scores = scores[cursor : cursor + batch_size]
        cursor += batch_size
        decisions_batch, diagnostics = ensemble.decision(batch_scores, images, tau)
        mean_threshold_log = diagnostics["thresholds_log"].mean(dim=0)
        margin = mean_threshold_log - diagnostics["scores_log"]
        margins.append(margin)
        decisions.append(decisions_batch)
        thresholds_log.append(diagnostics["thresholds_log"])
        thresholds_raw.append(diagnostics["thresholds_raw"])
        scores_raw.append(diagnostics["scores_raw"])
        scores_log.append(diagnostics["scores_log"])
        if "votes" in diagnostics:
            votes.append(diagnostics["votes"])
    stats = {
        "margin_log": torch.cat(margins) if margins else torch.tensor([]),
        "decisions": torch.cat(decisions) if decisions else torch.tensor([], dtype=torch.int),
        "thresholds_log": torch.cat(thresholds_log, dim=1) if thresholds_log else torch.empty(0),
        "thresholds_raw": torch.cat(thresholds_raw, dim=1) if thresholds_raw else torch.empty(0),
        "scores_raw": torch.cat(scores_raw) if scores_raw else torch.tensor([]),
        "scores_log": torch.cat(scores_log) if scores_log else torch.tensor([]),
        "member_label": member_label,
    }
    if votes:
        stats["votes"] = torch.cat(votes)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate quantile regression ensemble")
    parser.add_argument("--config", type=pathlib.Path, default=pathlib.Path("configs/attack_qr.yaml"))
    parser.add_argument("--data-config", type=pathlib.Path, default=pathlib.Path("configs/data_cifar10.yaml"))
    parser.add_argument("--ensemble", type=pathlib.Path, default=None, help="Optional path to bagging checkpoint")
    parser.add_argument("--fastdev", action="store_true", help="Eval on reduced samples for quick smoke tests")
    args = parser.parse_args()

    attack_cfg = load_yaml(args.config)
    data_cfg = load_yaml(args.data_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensemble_path = (
        args.ensemble
        if args.ensemble is not None
        else pathlib.Path(attack_cfg["logging"]["output_dir"]) / "ensembles" / "bagging.pt"
    )
    ensemble = BagOfQuantiles.load(ensemble_path, device=device)

    scores_cache = pathlib.Path(attack_cfg["t_error"]["cache_dir"])
    scores_in = torch.load(scores_cache / "eval_in.pt", map_location="cpu")["scores"]
    scores_out = torch.load(scores_cache / "eval_out.pt", map_location="cpu")["scores"]
    limit = None
    if args.fastdev:
        limit = 256
        scores_in = scores_in[:limit]
        scores_out = scores_out[:limit]

    indices_in = load_indices(pathlib.Path(data_cfg["splits"]["paths"]["eval_in"]))
    indices_out = load_indices(pathlib.Path(data_cfg["splits"]["paths"]["eval_out"]))

    loader_in = build_loader(data_cfg, "eval_in", indices_in, limit=limit)
    loader_out = build_loader(data_cfg, "eval_out", indices_out, limit=limit)

    metrics_by_tau = {}
    diagnostics = {}

    target_fprs = attack_cfg["target_fprs"]

    for tau in attack_cfg["bagging"]["tau_values"]:
        LOGGER.info("Evaluating tau=%.5f", tau)
        stats_in = compute_detection_stats(ensemble, loader_in, scores_in, tau, member_label=1)
        stats_out = compute_detection_stats(ensemble, loader_out, scores_out, tau, member_label=0)
        detections_in = stats_in["margin_log"]
        detections_out = stats_out["margin_log"]

        tau_metrics = {"roc_auc": roc_auc(detections_in, detections_out)}
        for fpr in target_fprs:
            tau_metrics[f"fpr_{fpr}"] = tpr_precision_at_fpr(
                detections_in,
                detections_out,
                target_fpr=fpr,
            )
        metrics_by_tau[tau] = tau_metrics
        diagnostics[tau] = {
            "stats_in": stats_in,
            "stats_out": stats_out,
        }

    run_id = time.strftime("%Y%m%d-%H%M%S")
    report_dir = pathlib.Path("reports") / run_id
    report_module.generate_report(
        report_dir,
        attack_cfg,
        data_cfg,
        metrics_by_tau,
        scores_in,
        scores_out,
        diagnostics,
        ensemble_path,
        args.config,
        args.data_config,
    )


if __name__ == "__main__":
    main()

