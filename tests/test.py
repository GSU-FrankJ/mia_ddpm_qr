# tests/test.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # 指向项目根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import torchvision.utils as vutils

from ddpm.engine.checkpoint_utils import load_ddpm_model
from ddpm.engine.sample_utils import sample
from ddpm.schedules.noise import DiffusionSchedule

device = "cuda"
model, schedule, meta = load_ddpm_model("runs/ddpm/cifar10_full/best.pt", device=device)
model.eval()
schedule = schedule.to(device)

with torch.no_grad():
    imgs = sample(model, schedule, shape=(64, 3, meta["img_size"], meta["img_size"]), device=device)
vutils.save_image(imgs, "runs/ddpm/cifar10_full/samples.png", nrow=8, normalize=True, value_range=(-1, 1))
