"""Evaluation metrics for QR-MIA."""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch

from mia_logging import get_winston_logger


LOGGER = get_winston_logger(__name__)


def _bootstrap(
    metric_fn: Callable[[torch.Tensor, torch.Tensor], float],
    scores_in: torch.Tensor,
    scores_out: torch.Tensor,
    num_samples: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    generator = torch.Generator().manual_seed(seed)
    values = []
    for _ in range(num_samples):
        idx_in = torch.randint(0, scores_in.numel(), (scores_in.numel(),), generator=generator)
        idx_out = torch.randint(0, scores_out.numel(), (scores_out.numel(),), generator=generator)
        values.append(metric_fn(scores_in[idx_in], scores_out[idx_out]))
    samples = torch.tensor(values)
    return torch.quantile(samples, 0.025).item(), torch.quantile(samples, 0.975).item()


def _precision(scores_in: torch.Tensor, scores_out: torch.Tensor, threshold: float) -> float:
    tp = (scores_in >= threshold).sum().item()
    fp = (scores_out >= threshold).sum().item()
    denom = tp + fp
    return tp / denom if denom > 0 else 0.0


def tpr_precision_at_fpr(
    scores_in: torch.Tensor,
    scores_out: torch.Tensor,
    target_fpr: float,
    num_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict:
    scores_in = scores_in.detach().cpu()
    scores_out = scores_out.detach().cpu()
    threshold = torch.quantile(scores_out, 1 - target_fpr).item()
    tpr = (scores_in >= threshold).float().mean().item()
    fpr = (scores_out >= threshold).float().mean().item()
    precision = _precision(scores_in, scores_out, threshold)

    def tpr_metric(si: torch.Tensor, so: torch.Tensor) -> float:
        thr = torch.quantile(so, 1 - target_fpr).item()
        return (si >= thr).float().mean().item()

    def precision_metric(si: torch.Tensor, so: torch.Tensor) -> float:
        thr = torch.quantile(so, 1 - target_fpr).item()
        return _precision(si, so, thr)

    tpr_ci = _bootstrap(tpr_metric, scores_in, scores_out, num_bootstrap, seed)
    precision_ci = _bootstrap(precision_metric, scores_in, scores_out, num_bootstrap, seed)

    LOGGER.info(
        "target_fpr=%.5f achieved_fpr=%.5f threshold=%.6f tpr=%.6f precision=%.6f",
        target_fpr,
        fpr,
        threshold,
        tpr,
        precision,
    )

    return {
        "threshold": threshold,
        "tpr": tpr,
        "tpr_ci": tpr_ci,
        "precision": precision,
        "precision_ci": precision_ci,
        "achieved_fpr": fpr,
        "fpr_error": fpr - target_fpr,
    }


def roc_auc(scores_in: torch.Tensor, scores_out: torch.Tensor) -> float:
    scores_in = scores_in.detach().cpu()
    scores_out = scores_out.detach().cpu()
    labels = torch.cat(
        [
            torch.ones_like(scores_in, dtype=torch.float32),
            torch.zeros_like(scores_out, dtype=torch.float32),
        ]
    )
    scores = torch.cat([scores_in, scores_out]).float()
    sorted_scores, indices = torch.sort(scores, descending=True)
    sorted_labels = labels[indices]

    tp = torch.cumsum(sorted_labels, dim=0)
    fp = torch.cumsum(1 - sorted_labels, dim=0)
    tp = torch.cat([torch.tensor([0.0]), tp])
    fp = torch.cat([torch.tensor([0.0]), fp])
    tpr = tp / tp[-1] if tp[-1] > 0 else tp
    fpr = fp / fp[-1] if fp[-1] > 0 else fp
    area = torch.trapz(tpr, fpr).item()
    LOGGER.info("ROC-AUC=%.6f", area)
    return area

