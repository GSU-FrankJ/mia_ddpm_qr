import torch

from attacks.eval.metrics import roc_auc, tpr_precision_at_fpr


def test_tpr_precision_basic():
    scores_in = torch.linspace(0.6, 1.0, steps=100)
    scores_out = torch.linspace(0.0, 0.4, steps=100)
    metrics = tpr_precision_at_fpr(scores_in, scores_out, target_fpr=0.01, num_bootstrap=50)
    assert metrics["tpr"] > 0.9
    assert metrics["precision"] > 0.9


def test_roc_auc_bounds():
    scores_in = torch.tensor([0.9, 0.8, 0.7])
    scores_out = torch.tensor([0.2, 0.3, 0.1])
    auc = roc_auc(scores_in, scores_out)
    assert 0.0 <= auc <= 1.0

