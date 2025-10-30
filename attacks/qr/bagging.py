"""Bagging ensemble for quantile regression thresholds."""

from __future__ import annotations

import pathlib
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Subset

from mia_logging import get_winston_logger
from attacks.qr.qr_dataset import QuantileRegressionDataset, train_val_split
from attacks.qr.qr_models import SmallCNNQuantile
from attacks.qr.qr_train import TrainConfig, train_quantile_model


LOGGER = get_winston_logger(__name__)


class BagOfQuantiles:
    """Bootstrap ensemble of quantile regressors.

    Majority voting is the default decision rule; averaged-threshold voting is
    available for ablations via ``vote="average"``.
    """

    def __init__(
        self,
        base_cfg: Dict,
        B: int = 50,
        bootstrap_ratio: float = 0.8,
        seed: int = 42,
        device: torch.device | None = None,
    ) -> None:
        self.base_cfg = base_cfg
        self.B = B
        self.bootstrap_ratio = bootstrap_ratio
        self.seed = seed
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models_by_tau: Dict[float, List[SmallCNNQuantile]] = {}
        self.histories_by_tau: Dict[float, List] = {}
        self.use_log1p: bool = self.base_cfg.get("log1p", True)

    def fit(
        self,
        scores_path: pathlib.Path,
        data_cfg: Dict,
        limit: int | None = None,
    ) -> None:
        dataset = QuantileRegressionDataset(data_cfg, scores_path, limit=limit)
        train_dataset, val_dataset = train_val_split(
            dataset,
            val_ratio=self.base_cfg.get("val_ratio", 0.1),
            seed=self.seed,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.base_cfg["batch_size"],
            shuffle=False,
            num_workers=data_cfg["dataset"].get("num_workers", 8),
        )

        generator = torch.Generator().manual_seed(self.seed)
        warmup_cfg = self.base_cfg.get("tau_warmup", {})
        warmup_value = warmup_cfg.get("value")
        warmup_epochs = warmup_cfg.get("epochs", 0)

        for tau in self.base_cfg.get("tau_values", [0.001]):
            self.models_by_tau[tau] = []
            self.histories_by_tau[tau] = []
            LOGGER.info("Training bagging ensemble for tau=%.5f", tau)
            for b in range(self.B):
                bootstrap_size = max(1, int(len(train_dataset) * self.bootstrap_ratio))
                bootstrap_indices = torch.randint(
                    low=0,
                    high=len(train_dataset),
                    size=(bootstrap_size,),
                    generator=generator,
                ).tolist()
                bootstrap_subset = Subset(train_dataset, bootstrap_indices)
                train_loader = DataLoader(
                    bootstrap_subset,
                    batch_size=self.base_cfg["batch_size"],
                    shuffle=True,
                    num_workers=data_cfg["dataset"].get("num_workers", 8),
                )
                model = SmallCNNQuantile()
                train_cfg = TrainConfig(
                    epochs=self.base_cfg["epochs"],
                    lr=self.base_cfg["lr"],
                    weight_decay=self.base_cfg.get("weight_decay", 0.0),
                    tau=tau,
                    device=self.device,
                    use_log1p=self.use_log1p,
                    warmup_tau=warmup_value if warmup_epochs > 0 else None,
                    warmup_epochs=warmup_epochs if warmup_epochs > 0 else 0,
                )
                result = train_quantile_model(model, train_loader, val_loader, train_cfg)
                fitted_model = SmallCNNQuantile()
                fitted_model.load_state_dict(result["state_dict"])
                fitted_model.to(self.device)
                fitted_model.eval()
                self.models_by_tau[tau].append(fitted_model)
                self.histories_by_tau[tau].append(result["history"])
                LOGGER.info("tau=%.5f model=%d best_val=%.6f", tau, b, result["best_val"])

    @torch.no_grad()
    def decision(
        self,
        scores: torch.Tensor,
        imgs: torch.Tensor,
        tau: float,
        vote: str = "majority",
    ) -> Tuple[torch.Tensor, Dict]:
        ensemble = self.models_by_tau.get(tau)
        if not ensemble:
            raise ValueError(f"No models trained for tau={tau}")

        imgs = imgs.to(self.device)
        scores_cpu = scores.detach().cpu().clamp_min(0)
        scores_log = torch.log1p(scores_cpu)
        thresholds_log = []
        for model in ensemble:
            preds = model(imgs)
            thresholds_log.append(preds.detach().cpu())
        thresholds_tensor = torch.stack(thresholds_log)
        thresholds_raw = torch.expm1(thresholds_tensor).clamp_min(0)

        if vote == "average":
            avg_thresholds = thresholds_tensor.mean(dim=0)
            decisions = (scores_log <= avg_thresholds).to(torch.int)
            diagnostics = {
                "thresholds_log": thresholds_tensor,
                "thresholds_raw": thresholds_raw,
                "avg_thresholds_log": avg_thresholds,
                "avg_thresholds_raw": torch.expm1(avg_thresholds).clamp_min(0),
                "scores_log": scores_log,
                "scores_raw": scores_cpu,
            }
            return decisions, diagnostics

        votes = (scores_log <= thresholds_tensor).sum(dim=0)
        decisions = (votes >= (len(ensemble) / 2)).to(torch.int)
        diagnostics = {
            "thresholds_log": thresholds_tensor,
            "thresholds_raw": thresholds_raw,
            "votes": votes,
            "scores_log": scores_log,
            "scores_raw": scores_cpu,
        }
        return decisions, diagnostics

    def save(self, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "B": self.B,
            "bootstrap_ratio": self.bootstrap_ratio,
            "seed": self.seed,
            "base_cfg": self.base_cfg,
            "state_dicts": {
                tau: [model.state_dict() for model in models]
                for tau, models in self.models_by_tau.items()
            },
        }
        torch.save(state, path)
        LOGGER.info("Saved bagging ensemble to %s", path)

    @classmethod
    def load(cls, path: pathlib.Path, device: torch.device | None = None) -> "BagOfQuantiles":
        state = torch.load(path, map_location=device or "cpu")
        instance = cls(
            base_cfg=state["base_cfg"],
            B=state["B"],
            bootstrap_ratio=state["bootstrap_ratio"],
            seed=state["seed"],
            device=device,
        )
        for tau, models_state in state["state_dicts"].items():
            instance.models_by_tau[float(tau)] = []
            for sd in models_state:
                model = SmallCNNQuantile()
                model.load_state_dict(sd)
                model.to(instance.device)
                model.eval()
                instance.models_by_tau[float(tau)].append(model)
        return instance

