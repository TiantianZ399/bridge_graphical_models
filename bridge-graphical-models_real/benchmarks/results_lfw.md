# LFW subset benchmark results

These numbers were generated in the execution environment with:

```bash
PYTHONPATH=src python -B scripts/run_realdata_gap.py \
  --dataset lfw --n 150 --seeds 0 1 2 --k-neighbors 16 \
  --out-dir outputs/realdata_gap

PYTHONPATH=src python -B scripts/run_realdata_benchmark.py \
  --dataset lfw --n 150 --seeds 0 1 2 --k-neighbors 16 \
  --ridge-repeats 8 --out-dir outputs/realdata_benchmark
```

The LFW subset is a no-download real-image dataset bundled with scikit-image. Images are embedded using 16-dimensional PCA-whitened latents; the PCA components explain about 89.8% of training variance.

| Dataset | Coupling | Gap AUC ↓ | Eval velocity MSE ↓ | Latent MMD² ↓ |
|---|---|---:|---:|---:|
| LFW subset | independent | 23.701 ± 0.880 | 23.511 ± 1.385 | 0.02780 ± 0.00649 |
| LFW subset | minibatch OT | 11.871 ± 0.201 | 12.877 ± 0.078 | 0.02744 ± 0.00686 |

Interpretation: minibatch OT roughly halves the Markovization gap and reduces the fixed-model velocity-regression error by about 45% under the same bridge and regression model. Latent MMD² is nearly unchanged in this tiny benchmark, so this should be reported as evidence for the gap/training-difficulty link, not as a sample-quality claim.
