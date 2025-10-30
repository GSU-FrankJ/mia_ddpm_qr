"""Report generation for QR-MIA experiments."""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
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
    arr_a = np.asarray(data_a).ravel()
    arr_b = np.asarray(data_b).ravel()
    combined = np.concatenate([arr_a, arr_b]) if (arr_a.size or arr_b.size) else np.array([])
    bins = 50
    path.parent.mkdir(parents=True, exist_ok=True)
    if combined.size == 0 or np.allclose(combined.max(), combined.min()):
        plt.figure(figsize=(6, 4))
        counts = [arr_a.size, arr_b.size]
        plt.bar([0, 1], counts, tick_label=[label_a, label_b], alpha=0.7)
        plt.ylabel("Count")
        plt.title(title + " (degenerate)")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return

    plt.figure(figsize=(8, 5))
    plt.hist(arr_a, bins=bins, alpha=0.6, label=label_a)
    plt.hist(arr_b, bins=bins, alpha=0.6, label=label_b)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
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
        lines.append("| Target FPR | TPR | 95% CI | Precision | 95% CI | Achieved FPR | FPR Error |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for key, value in metrics.items():
            if key == "roc_auc":
                continue
            target = key.split("_")[-1]
            lines.append(
                f"| {target} | {value['tpr']:.6f} | ({value['tpr_ci'][0]:.6f}, {value['tpr_ci'][1]:.6f}) "
                f"| {value['precision']:.6f} | ({value['precision_ci'][0]:.6f}, {value['precision_ci'][1]:.6f}) "
                f"| {value['achieved_fpr']:.6f} | {value['fpr_error']:.6f} |"
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
        thresholds_in_raw = diag["stats_in"]["thresholds_raw"].cpu().numpy()
        thresholds_out_raw = diag["stats_out"]["thresholds_raw"].cpu().numpy()
        threshold_hist = report_dir / f"thresholds_tau_{tau}.png"
        _histogram(
            thresholds_in_raw.flatten(),
            thresholds_out_raw.flatten(),
            threshold_hist,
            "in thresholds",
            "out thresholds",
            f"Threshold Distribution (tau={tau})",
        )
        roc_path = report_dir / f"roc_tau_{tau}.png"
        fpr, tpr = _roc_curve(
            diag["stats_in"]["margin_log"].detach().cpu(),
            diag["stats_out"]["margin_log"].detach().cpu(),
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

            stats_in = diag["stats_in"]
            stats_out = diag["stats_out"]
            thresholds_in_log = stats_in["thresholds_log"].T.cpu().numpy()
            thresholds_in_raw = stats_in["thresholds_raw"].T.cpu().numpy()
            thresholds_out_log = stats_out["thresholds_log"].T.cpu().numpy()
            thresholds_out_raw = stats_out["thresholds_raw"].T.cpu().numpy()

            member_label_in = int(stats_in["member_label"])
            member_label_out = int(stats_out["member_label"])

            df_in = pd.DataFrame(
                {
                    "score_raw": stats_in["scores_raw"].cpu().numpy(),
                    "score_log": stats_in["scores_log"].cpu().numpy(),
                    "margin_log": stats_in["margin_log"].cpu().numpy(),
                    "final_decision": stats_in["decisions"].cpu().numpy(),
                    "member_label": member_label_in,
                    "thresholds_per_model_raw": [list(row) for row in thresholds_in_raw],
                    "thresholds_per_model_log": [list(row) for row in thresholds_in_log],
                }
            )
            votes_in = stats_in.get("votes")
            if votes_in is not None:
                df_in["votes"] = votes_in.cpu().numpy()

            df_out = pd.DataFrame(
                {
                    "score_raw": stats_out["scores_raw"].cpu().numpy(),
                    "score_log": stats_out["scores_log"].cpu().numpy(),
                    "margin_log": stats_out["margin_log"].cpu().numpy(),
                    "final_decision": stats_out["decisions"].cpu().numpy(),
                    "member_label": member_label_out,
                    "thresholds_per_model_raw": [list(row) for row in thresholds_out_raw],
                    "thresholds_per_model_log": [list(row) for row in thresholds_out_log],
                }
            )
            votes_out = stats_out.get("votes")
            if votes_out is not None:
                df_out["votes"] = votes_out.cpu().numpy()
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

    report_payload = {
        "seed": attack_cfg.get("seed"),
        "target_fprs": attack_cfg.get("target_fprs", []),
        "t_error": {
            "T": attack_cfg.get("t_error", {}).get("T"),
            "k_uniform": attack_cfg.get("t_error", {}).get("k_uniform"),
            "aggregate": attack_cfg.get("t_error", {}).get("aggregate"),
        },
        "bagging": {
            "B": attack_cfg.get("bagging", {}).get("B"),
            "bootstrap_ratio": attack_cfg.get("bagging", {}).get("bootstrap_ratio"),
        },
        "hashes": {
            "attack_config": _hash_file(pathlib.Path(attack_config_path)),
            "data_config": _hash_file(pathlib.Path(data_config_path)),
            "ensemble": _hash_file(ensemble_path) if ensemble_path.is_file() else None,
        },
        "results": [],
    }

    for tau, metrics in metrics_by_tau.items():
        tau_entry = {
            "tau": float(tau),
            "roc_auc": metrics.get("roc_auc"),
            "metrics": [],
        }
        for key, value in metrics.items():
            if key == "roc_auc":
                continue
            target = key.split("_")[-1]
            tau_entry["metrics"].append(
                {
                    "target_fpr": float(target),
                    "threshold": value["threshold"],
                    "tpr": value["tpr"],
                    "precision": value["precision"],
                    "achieved_fpr": value["achieved_fpr"],
                    "fpr_calibration_error": value["fpr_error"],
                    "ci_95": {
                        "tpr": [float(value["tpr_ci"][0]), float(value["tpr_ci"][1])],
                        "precision": [
                            float(value["precision_ci"][0]),
                            float(value["precision_ci"][1]),
                        ],
                    },
                }
            )
        report_payload["results"].append(tau_entry)

    with (report_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report_payload, handle, indent=2)

