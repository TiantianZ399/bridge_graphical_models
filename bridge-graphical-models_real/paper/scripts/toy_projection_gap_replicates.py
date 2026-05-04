"""Replicate the Markovization-gap toy diagnostic.

This script compares independent endpoint pairing with minibatch OT pairing
for a straight-line bridge between a standard normal base and an 8-Gaussian
ring target. It estimates E Tr Var(U | X_t) with k-nearest-neighbor conditional
means and integrates over t.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def sample_ring_gaussians(n: int, k: int = 8, radius: float = 3.0,
                          noise: float = 0.25, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = np.array([
        [radius * np.cos(2 * np.pi * i / k), radius * np.sin(2 * np.pi * i / k)]
        for i in range(k)
    ])
    comp = rng.integers(0, k, size=n)
    return centers[comp] + noise * rng.normal(size=(n, 2))


def knn_gap(z: np.ndarray, x: np.ndarray, ts: np.ndarray, k_neighbors: int = 32) -> np.ndarray:
    u = x - z
    gaps = []
    for t in ts:
        xt = (1 - t) * z + t * x
        idx = NearestNeighbors(n_neighbors=k_neighbors).fit(xt).kneighbors(xt, return_distance=False)
        mean_u = u[idx].mean(axis=1)
        gaps.append(np.mean(np.sum((u - mean_u) ** 2, axis=1)))
    return np.asarray(gaps)


def run_one(n: int, seed: int, k_neighbors: int = 32):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 2))
    x = sample_ring_gaussians(n, seed=seed + 100)
    ts = np.linspace(0.05, 0.95, 19)

    gap_ind = knn_gap(z, x, ts, k_neighbors=k_neighbors)

    cost = np.sum((z[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    row, col = linear_sum_assignment(cost)
    x_ot = np.empty_like(x)
    x_ot[row] = x[col]
    gap_ot = knn_gap(z, x_ot, ts, k_neighbors=k_neighbors)

    return np.trapezoid(gap_ind, ts), np.trapezoid(gap_ot, ts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=Path, default=Path('figures'))
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--k-neighbors', type=int, default=32)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for n in (500, 1000):
        for seed in range(args.seeds):
            ind, ot = run_one(n, seed, args.k_neighbors)
            rows.append((n, seed, ind, ot))
            print(f'n={n} seed={seed} independent={ind:.6f} ot={ot:.6f}')

    data = np.array(rows, dtype=float)
    np.savetxt(args.out_dir / 'toy_gap_replicates.csv', data, delimiter=',',
               header='n,seed,independent_auc,ot_auc', comments='')

    fig, ax = plt.subplots(figsize=(4.4, 2.4))
    xs = np.array([0, 1])
    width = 0.35
    labels = []
    means_ind = []
    stds_ind = []
    means_ot = []
    stds_ot = []
    for n in (500, 1000):
        subset = data[data[:, 0] == n]
        labels.append(str(n))
        means_ind.append(subset[:, 2].mean())
        stds_ind.append(subset[:, 2].std(ddof=1))
        means_ot.append(subset[:, 3].mean())
        stds_ot.append(subset[:, 3].std(ddof=1))
    ax.bar(xs - width/2, means_ind, width, yerr=stds_ind, capsize=3, label='independent')
    ax.bar(xs + width/2, means_ot, width, yerr=stds_ot, capsize=3, label='minibatch OT')
    ax.set_yscale('log')
    ax.set_xticks(xs, labels)
    ax.set_xlabel('sample size n')
    ax.set_ylabel('integrated Markovization gap')
    ax.set_title('Gap reduction is robust across seeds')
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_dir / 'toy_gap_replicates.pdf', bbox_inches='tight')
    fig.savefig(args.out_dir / 'toy_gap_replicates.png', dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    main()
