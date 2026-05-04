# Real-data latent benchmarks

The benchmark suite tests the paper's operational claim:

> coupling/bridge choice should change the Markovization gap, and lower gap should correlate with easier latent flow-matching training under fixed compute.

The default benchmark uses the scikit-image `lfw` subset, which requires no download. Standard image datasets use torchvision and can be downloaded locally.

## Gap-only benchmark

```bash
PYTHONPATH=src python scripts/run_realdata_gap.py --dataset lfw --n 150 --seeds 0 1 2
PYTHONPATH=src python scripts/run_realdata_gap.py --dataset mnist --download --n 2000 --seeds 0 1 2
PYTHONPATH=src python scripts/run_realdata_gap.py --dataset fashionmnist --download --n 2000 --seeds 0 1 2
PYTHONPATH=src python scripts/run_realdata_gap.py --dataset cifar10 --download --latent-dim 32 --n 2000 --seeds 0 1 2
```

Outputs are written under `outputs/realdata_gap/<dataset>/`.

## Train-and-sample latent benchmark

```bash
PYTHONPATH=src python scripts/run_realdata_benchmark.py --dataset lfw --n 150 --ridge-repeats 8 --seeds 0 1 2
```

For standard datasets:

```bash
PYTHONPATH=src python scripts/run_realdata_benchmark.py --dataset mnist --download --n 2000 --ridge-repeats 8 --seeds 0 1 2
PYTHONPATH=src python scripts/run_realdata_benchmark.py --dataset fashionmnist --download --n 2000 --ridge-repeats 8 --seeds 0 1 2
PYTHONPATH=src python scripts/run_realdata_benchmark.py --dataset cifar10 --download --latent-dim 32 --n 2000 --ridge-repeats 8 --seeds 0 1 2
```

Metrics:

- `gap_auc`: integrated k-NN Markovization-gap estimate.
- `train_mse`: ridge velocity-regression MSE on sampled bridge times.
- `eval_velocity_mse`: velocity MSE on fresh bridge times.
- `latent_mmd2`: RBF MMD^2 between generated and held-out target latents.
- `latent_mean_error`, `latent_cov_error`: first- and second-moment mismatch.

This is not a state-of-the-art image benchmark. It is a controlled latent benchmark designed to test whether the BGM diagnostic predicts training behavior.
