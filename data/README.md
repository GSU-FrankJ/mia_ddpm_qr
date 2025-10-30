# Data Directory

This directory contains dataset files and split indices for reproducible experiments.

## Structure

```
data/
├── splits/          # JSON files with image indices for each split
│   ├── aux.json     # Indices for QR training (10k test set)
│   ├── eval_in.json # Positive samples (5k from member_train)
│   └── eval_out.json # Negative samples (5k from remaining train)
└── cifar10/         # CIFAR-10 dataset (auto-downloaded)
```

## Split Protocol

### Overview
We use a strict split protocol to ensure no data leakage:

1. **member_train** (40k): Used to train the DDIM model
2. **eval_in** (5k): Sampled from member_train, excluded from QR training
3. **eval_out** (5k): From remaining 10k train images (never used to train DDIM)
4. **aux** (10k): Full test set, used ONLY for QR training/calibration

### Why This Split Avoids Leakage

- **DDIM training**: Only sees `member_train` (40k images)
- **QR training**: Only sees `aux` (10k test set), never sees `eval_in` or `eval_out`
- **MIA evaluation**: Tests on `eval_in` (members) vs `eval_out` (non-members)
- **No overlap**: Each split is disjoint, ensuring fair evaluation

### Generating Splits

Run the split script to generate reproducible splits:

```bash
bash scripts/split_cifar10.sh
```

This creates the JSON files in `data/splits/` with deterministic indices based on the seed in `configs/data_cifar10.yaml`.

