#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from attack_qr.engine.eval_attack import EvalConfig, evaluate_attack, load_quantile_ensemble
from ddpm.data.loader import load_split_indices
from ddpm.engine.checkpoint_utils import load_ddpm_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate QR-MIA attack.")
    parser.add_argument("--config", type=str, required=True, help="Evaluation YAML config.")
    parser.add_argument("--split_json", type=str, required=True, help="Split JSON path.")
    parser.add_argument("--ddpm_ckpt", type=str, required=True, help="Trained DDPM checkpoint.")
    parser.add_argument("--qr_dir", type=str, required=True, help="Directory with trained quantile models.")
    parser.add_argument("--report", type=str, required=True, help="Path to write final report JSON.")
    parser.add_argument("--alpha", type=float, default=None, help="Override alpha for evaluation.")
    parser.add_argument("--mode", type=str, default=None, help="Override t-error mode.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device.")
    parser.add_argument("--data-root", type=str, default="data", help="Dataset root.")
    parser.add_argument("--seed", type=int, default=None, help="Override seed.")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    eval_cfg = config.get("eval", {})
    seed = args.seed if args.seed is not None else eval_cfg.get("seed", config.get("seed", 0))
    alpha = args.alpha if args.alpha is not None else eval_cfg.get("alpha", 0.01)
    mode = args.mode if args.mode is not None else eval_cfg.get("mode", "x0")
    eval_config = EvalConfig(
        alpha=alpha,
        mode=mode,
        K=eval_cfg.get("K", 4),
        batch_size=eval_cfg.get("batch_size", 128),
        bootstrap=eval_cfg.get("bootstrap", 200),
        seed=seed,
    )

    device = torch.device(args.device)
    ddpm_model, schedule, ddpm_meta = load_ddpm_model(Path(args.ddpm_ckpt), device=device)
    dataset_name = ddpm_meta["dataset"]
    img_size = ddpm_meta["img_size"]

    ensemble, alpha_list = load_quantile_ensemble(Path(args.qr_dir), device=device)

    split = load_split_indices(args.split_json)
    z_indices = split["z"]
    holdout_indices = split["holdout"]
    rng = np.random.default_rng(seed)
    member_indices = rng.choice(z_indices, size=len(holdout_indices), replace=False).tolist()

    out_dir = Path(args.report).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    report = evaluate_attack(
        ddpm_model=ddpm_model,
        schedule=schedule,
        ensemble=ensemble,
        alpha_list=alpha_list,
        config=eval_config,
        dataset_name=dataset_name,
        data_root=args.data_root,
        member_indices=member_indices,
        nonmember_indices=holdout_indices,
        img_size=img_size,
        global_seed=seed,
        device=device,
        out_dir=out_dir,
    )

    report_path = Path(args.report)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Evaluation report written to {report_path}")


if __name__ == "__main__":
    main()
