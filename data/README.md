# CIFAR-10 Splits

This directory stores reproducible membership inference splits for CIFAR-10.

The `splits` subdirectory is populated by `scripts/split_cifar10.sh`, which
invokes the Python data-splitting utility with a deterministic seed. All JSON
files contain arrays of image indices relative to the canonical CIFAR-10
Training (50,000 images) or Test (10,000 images) partitions.

- `member_train.json`: 40,000 indices used to train the DDIM. These include the
  5,000 images allocated to `eval_in`, maintaining parity between training and
  evaluation positives.
- `aux.json`: 10,000 indices drawn from the CIFAR-10 test **only**; reserved
  exclusively for quantile regression calibration and never used to train the
  diffusion model. This guardrail prevents leakage between the attack model
  and the target generator.
- `eval_in.json`: 5,000 indices sampled from the 40,000 images that actually
  train the DDIM. We remove these indices from quantile training to simulate
  the classic in-distribution membership scenario.
- `eval_out.json`: 5,000 indices drawn from the 10,000 CIFAR-10 training
  images that were intentionally withheld from DDIM optimisation, ensuring
  they are definitively non-members.

The deterministic procedure ensures that every experiment, even across
machines, uses the exact same membership / non-membership assignments.

