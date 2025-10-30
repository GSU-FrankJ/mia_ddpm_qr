import torch

from attacks.scores.t_error import t_error_aggregate, uniform_timesteps


class ZeroModel(torch.nn.Module):
    def forward(self, x, t):  # type: ignore[override]
        return torch.zeros_like(x)


def test_t_error_shape():
    model = ZeroModel()
    x0 = torch.randn(4, 3, 32, 32)
    alphas_bar = torch.linspace(0.1, 0.99, steps=1000)
    timesteps = uniform_timesteps(1000, 5)
    scores = t_error_aggregate(x0, timesteps, model, alphas_bar, agg="mean")
    assert scores.shape == (4,)


def test_uniform_timesteps_bounds():
    timesteps = uniform_timesteps(100, 10)
    assert timesteps[0] >= 0 and timesteps[-1] <= 99
    assert len(set(timesteps)) == 10


