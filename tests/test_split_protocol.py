import numpy as np

from ddpm.data.split import stratified_split


def test_splits_disjoint_and_complete():
    labels = np.array([i // 10 for i in range(100)])
    splits = stratified_split(labels, seed=0)
    sets = [set(splits[name]) for name in ["Z", "Public", "Holdout"]]
    assert len(sets[0].intersection(sets[1])) == 0
    assert len(sets[0].intersection(sets[2])) == 0
    assert len(sets[1].intersection(sets[2])) == 0
    union = set().union(*sets)
    assert union == set(range(len(labels)))


def test_class_balance_within_tolerance():
    counts = [101, 99, 120]
    labels = np.concatenate([np.full(c, idx) for idx, c in enumerate(counts)])
    splits = stratified_split(labels, seed=42)
    for cls, total in enumerate(counts):
        cls_indices = np.where(labels == cls)[0]
        expected = {
            "Z": 0.5 * total,
            "Public": 0.25 * total,
            "Holdout": 0.25 * total,
        }
        for split_name in ["Z", "Public", "Holdout"]:
            actual = len(set(cls_indices).intersection(splits[split_name]))
            tolerance = max(1, int(0.01 * total))
            assert abs(actual - expected[split_name]) <= tolerance
