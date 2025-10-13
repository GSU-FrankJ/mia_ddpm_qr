import torch

from attack_qr.utils.losses import pinball_loss


def test_pinball_loss_scalar():
    pred = torch.tensor([0.0, 1.0, 2.0])
    target = torch.tensor([1.0, 1.0, 1.0])
    loss = pinball_loss(pred, target, alpha=0.5, reduction="none")
    expected = torch.tensor([0.5, 0.0, 0.5])
    assert torch.allclose(loss, expected)


def test_pinball_loss_zero_alpha_limits():
    pred = torch.tensor([[0.0, 1.0]])
    target = torch.tensor([[1.0, 1.0]])
    alpha = torch.tensor([0.1, 0.9])
    loss = pinball_loss(pred, target, alpha=alpha, reduction="mean")
    manual = torch.maximum(alpha * (target - pred), (alpha - 1) * (target - pred))
    assert torch.isclose(loss, manual.mean())


def test_pinball_loss_shapes():
    pred = torch.zeros(4, 2)
    target = torch.ones(4).unsqueeze(1)
    alpha = [0.1, 0.9]
    loss = pinball_loss(pred, target.expand(-1, 2), alpha=alpha, reduction="none")
    assert loss.shape == pred.shape
