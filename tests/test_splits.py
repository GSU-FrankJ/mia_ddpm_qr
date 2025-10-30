from scripts.split_cifar10 import build_splits
import yaml


def test_split_reproducibility(tmp_path):
    cfg = yaml.safe_load(
        """
dataset:
  root: data/cifar10
splits:
  seed: 12345
        """
    )
    first = build_splits(cfg)
    second = build_splits(cfg)
    assert first == second
    assert len(first["member_train"]) == 40000
    assert set(first["eval_in"]).issubset(set(first["member_train"]))


