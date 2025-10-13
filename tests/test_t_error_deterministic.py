import torch

from attack_qr.features.t_error import compute_t_error
from ddpm.schedules.noise import DiffusionSchedule


class DummyModel(torch.nn.Module):
    def forward(self, x, t):
        return torch.zeros_like(x)


def test_t_error_determinism_same_seed():
    model = DummyModel()
    schedule = DiffusionSchedule(T=10, beta_schedule="linear")
    x0 = torch.ones(1, 3, 4, 4)
    t = torch.tensor([3], dtype=torch.long)
    idx = [42]
    seed = 123
    r1 = compute_t_error(model, schedule, x0, t, "test", idx, seed, mode="eps")
    r2 = compute_t_error(model, schedule, x0, t, "test", idx, seed, mode="eps")
    assert torch.allclose(r1, r2)


def test_t_error_changes_with_seed():
    model = DummyModel()
    schedule = DiffusionSchedule(T=10, beta_schedule="linear")
    x0 = torch.ones(1, 3, 4, 4)
    t = torch.tensor([5], dtype=torch.long)
    idx = [7]
    r1 = compute_t_error(model, schedule, x0, t, "test", idx, 1, mode="eps")
    r2 = compute_t_error(model, schedule, x0, t, "test", idx, 2, mode="eps")
    assert not torch.allclose(r1, r2)
