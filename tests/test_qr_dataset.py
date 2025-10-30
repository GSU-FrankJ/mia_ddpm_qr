import json
import torch

from attacks.qr import qr_dataset


def test_quantile_dataset_alignment(monkeypatch, tmp_path):
    aux_indices = list(range(10))
    aux_json = tmp_path / "aux.json"
    aux_json.write_text(json.dumps(aux_indices), encoding="utf-8")
    scores_path = tmp_path / "scores.pt"
    torch.save({"scores": torch.randn(10)}, scores_path)

    class DummyDataset:
        def __init__(self, *args, **kwargs):
            self.data = [torch.zeros(3, 32, 32) for _ in aux_indices]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], 0

    monkeypatch.setattr(qr_dataset, "SplitDataset", DummyDataset)

    cfg = {
        "splits": {"paths": {"aux": str(aux_json)}},
        "dataset": {"root": "", "normalization": {"mean": [0, 0, 0], "std": [1, 1, 1]}},
    }
    dataset = qr_dataset.QuantileRegressionDataset(cfg, scores_path)
    assert len(dataset) == 10
    image, score_raw, score_log = dataset[0]
    assert image.shape == (3, 32, 32)
    assert isinstance(score_raw.item(), float)
    assert torch.allclose(score_log, torch.log1p(score_raw.clamp_min(0)))


