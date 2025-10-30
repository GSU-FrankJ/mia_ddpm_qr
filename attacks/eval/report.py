"""Report generation for QR-MIA experiments."""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Dict

import matplotlib.pyplot as plt
import torch

from mia_logging import get_winston_logger


LOGGER = get_winston_logger(__name__)


def _hash_file(path: pathlib.Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8192)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _histogram(data_a, data_b, path: pathlib.Path, label_a: str, label_b: str, title: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(data_a, bins=50, alpha=0.6, label=label_a)
    plt.hist(data_b, bins=50, alpha=0.6, label=label_b)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def _roc_curve(scores_in: torch.Tensor, scores_out: torch.Tensor):
    labels = torch.cat([
        torch.ones_like(scores_in, dtype=torch.float32),
        torch.zeros_like(scores_out, dtype=torch.float32),
    ])
    scores = torch.cat([scores_in, scores_out]).float()
    sorted_idx = torch.argsort(scores, descending=True)
    sorted_labels = labels[sorted_idx]
    tp = torch.cumsum(sorted_labels, dim=0)
    fp = torch.cumsum(1 - sorted_labels, dim=0)
    tp = torch.cat([torch.tensor([0.0]), tp])
    fp = torch.cat([torch.tensor([0.0]), fp])
    tpr = tp / tp[-1] if tp[-1] > 0 else tp
    fpr = fp / fp[-1] if fp[-1] > 0 else fp
    return fpr.numpy(), tpr.numpy()


def _write_markdown(
    report_path: pathlib.Path,
    run_id: str,
    attack_config_path: pathlib.Path,
    data_config_path: pathlib.Path,
    ensemble_path: pathlib.Path,
    metrics_by_tau: Dict,
) -> None:
    lines = ["# QR-MIA Evaluation Report", ""]
    lines.append(f"- Run ID: `{run_id}`")
    lines.append(f"- Attack config: `{attack_config_path}` (sha1 `{_hash_file(attack_config_path)}`)")
    lines.append(f"- Data config: `{data_config_path}` (sha1 `{_hash_file(data_config_path)}`)")
    lines.append(f"- Ensemble checkpoint: `{ensemble_path}`")
    lines.append("")

    for tau, metrics in metrics_by_tau.items():
        lines.append(f"## Tau = {tau}")
        lines.append("")
        lines.append("| Target FPR | TPR | 95% CI | Precision | 95% CI | Achieved FPR |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for key, value in metrics.items():
            if key == "roc_auc":
                continue
            target = key.split("_")[-1]
            lines.append(
                f"| {target} | {value['tpr']:.6f} | ({value['tpr_ci'][0]:.6f}, {value['tpr_ci'][1]:.6f}) "
                f"| {value['precision']:.6f} | ({value['precision_ci'][0]:.6f}, {value['precision_ci'][1]:.6f}) "
                f"| {value['achieved_fpr']:.6f} |"
            )
        lines.append("")
        lines.append(f"ROC-AUC: **{metrics['roc_auc']:.6f}**")
        lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Wrote markdown report to %s", report_path)


def generate_report(
    report_dir: pathlib.Path,
    attack_cfg: Dict,
    data_cfg: Dict,
    metrics_by_tau: Dict,
    scores_in: torch.Tensor,
    scores_out: torch.Tensor,
    diagnostics: Dict,
    ensemble_path: pathlib.Path,
    attack_config_path: pathlib.Path,
    data_config_path: pathlib.Path,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)

    metrics_json = report_dir / "metrics.json"
    with metrics_json.open("w", encoding="utf-8") as handle:
        json.dump(metrics_by_tau, handle, indent=2)

    score_hist = report_dir / "scores_hist.png"
    _histogram(
        scores_in.numpy(),
        scores_out.numpy(),
        score_hist,
        "eval_in",
        "eval_out",
        "t-error Scores",
    )

    for tau, diag in diagnostics.items():
        thresholds = diag["stats_in"]["thresholds"].flatten().cpu().numpy()
        threshold_hist = report_dir / f"thresholds_tau_{tau}.png"
        _histogram(
            thresholds,
            diag["stats_out"]["thresholds"].flatten().cpu().numpy(),
            threshold_hist,
            "in thresholds",
            "out thresholds",
            f"Threshold Distribution (tau={tau})",
        )
        roc_path = report_dir / f"roc_tau_{tau}.png"
        fpr, tpr = _roc_curve(
            diag["stats_in"]["stat"].detach().cpu(),
            diag["stats_out"]["stat"].detach().cpu(),
        )
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label="ROC")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"ROC Curve tau={tau}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(roc_path)
        plt.close()

        try:
            import pandas as pd  # type: ignore

            df_in = pd.DataFrame(
                {
                    "score": diag["stats_in"]["scores"].cpu().numpy(),
                    "stat": diag["stats_in"]["stat"].cpu().numpy(),
                    "decision": diag["stats_in"]["decisions"].cpu().numpy(),
                    "avg_threshold": diag["stats_in"]["thresholds"].mean(dim=0).cpu().numpy(),
                }
            )
            df_out = pd.DataFrame(
                {
                    "score": diag["stats_out"]["scores"].cpu().numpy(),
                    "stat": diag["stats_out"]["stat"].cpu().numpy(),
                    "decision": diag["stats_out"]["decisions"].cpu().numpy(),
                    "avg_threshold": diag["stats_out"]["thresholds"].mean(dim=0).cpu().numpy(),
                }
            )
            df_in.to_parquet(report_dir / f"diagnostics_in_tau_{tau}.parquet")
            df_out.to_parquet(report_dir / f"diagnostics_out_tau_{tau}.parquet")
        except ImportError:
            LOGGER.warning("pandas/pyarrow not available; skipping parquet diagnostics")

    _write_markdown(
        report_dir / "report.md",
        report_dir.name,
        attack_config_path,
        data_config_path,
        ensemble_path,
        metrics_by_tau,
    )

