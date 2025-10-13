from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.metrics import auc, roc_auc_score, roc_curve


@dataclass
class RocMetrics:
    fprs: np.ndarray
    tprs: np.ndarray
    thresholds: np.ndarray
    auc: float


def compute_roc(labels: np.ndarray, scores: np.ndarray) -> RocMetrics:
    fprs, tprs, thresholds = roc_curve(labels, scores)
    return RocMetrics(fprs=fprs, tprs=tprs, thresholds=thresholds, auc=roc_auc_score(labels, scores))


def interpolate_tpr(fprs: np.ndarray, tprs: np.ndarray, target_fpr: float) -> float:
    if target_fpr <= fprs[0]:
        return tprs[0]
    if target_fpr >= fprs[-1]:
        return tprs[-1]
    return np.interp(target_fpr, fprs, tprs)


def bootstrap_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    target_fprs: Iterable[float],
    n_bootstrap: int,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    target_fprs = list(target_fprs)
    auc_samples: List[float] = []
    tpr_samples: Dict[float, List[float]] = {fpr: [] for fpr in target_fprs}

    n = len(labels)
    indices = np.arange(n)
    for _ in range(n_bootstrap):
        sample_idx = rng.choice(indices, size=n, replace=True)
        roc = compute_roc(labels[sample_idx], scores[sample_idx])
        auc_samples.append(roc.auc)
        for fpr in target_fprs:
            tpr_samples[fpr].append(interpolate_tpr(roc.fprs, roc.tprs, fpr))

    summary: Dict[str, Dict[str, float]] = {"auc": summary_stats(np.array(auc_samples))}
    for fpr in target_fprs:
        summary[f"tpr@{fpr}"] = summary_stats(np.array(tpr_samples[fpr]))
    return summary


def summary_stats(samples: np.ndarray, alpha: float = 0.95) -> Dict[str, float]:
    lower = np.quantile(samples, (1.0 - alpha) / 2.0)
    upper = np.quantile(samples, 1.0 - (1.0 - alpha) / 2.0)
    return {
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples, ddof=1)),
        "ci_low": float(lower),
        "ci_high": float(upper),
    }

