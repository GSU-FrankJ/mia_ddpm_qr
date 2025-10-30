import torch

from attacks.qr.bagging import BagOfQuantiles


class ConstantModel(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(value), requires_grad=False)

    def forward(self, x):  # type: ignore[override]
        batch = x.shape[0]
        return self.value.expand(batch)


def test_bagging_majority_vote():
    bag = BagOfQuantiles(base_cfg={"batch_size": 1, "epochs": 1, "lr": 1e-3, "val_ratio": 0.1, "tau_values": [0.1]}, B=3)
    bag.models_by_tau[0.1] = [ConstantModel(0.45), ConstantModel(0.35), ConstantModel(0.65)]
    scores = torch.tensor([0.5, 0.7])
    imgs = torch.zeros(2, 3, 32, 32)
    decisions, diagnostics = bag.decision(scores, imgs, tau=0.1)
    assert decisions.tolist() == [1, 0]
    assert diagnostics["thresholds_log"].shape == (3, 2)
    assert torch.allclose(torch.expm1(diagnostics["thresholds_log"]), diagnostics["thresholds_raw"], atol=1e-6)
    assert torch.allclose(diagnostics["scores_log"], torch.log1p(scores))


