import torch

from attacks.qr.qr_models import pinball_loss


def test_pinball_loss_monotonicity():
    pred = torch.tensor([0.0, 0.5, 1.0], requires_grad=True)
    target = torch.zeros_like(pred)
    loss = pinball_loss(pred, target, tau=0.1)
    loss.backward()
    assert pred.grad[0] < pred.grad[2]


def test_pinball_loss_zero_residual():
    pred = torch.ones(5)
    target = torch.ones(5)
    loss = pinball_loss(pred, target, tau=0.5)
    assert torch.isclose(loss, torch.tensor(0.0))

