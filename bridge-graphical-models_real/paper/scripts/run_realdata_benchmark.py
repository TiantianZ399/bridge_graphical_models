#!/usr/bin/env python
"""Train and evaluate latent straight-bridge models on real datasets.

The default ``lfw`` run requires no network access. MNIST/Fashion-MNIST/CIFAR-10
can be run locally with ``--download``. The default learner is a pure-NumPy ridge
velocity model; this keeps the benchmark lightweight and deterministic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from psvb.couplings import independent_pairing, minibatch_ot_pairing
from psvb.gap import estimate_markovization_gap
from psvb.linear_training import fit_linear_velocity_model, sample_linear_model
from psvb.metrics import mean_cov_error, rbf_mmd2
from psvb.realdata import load_image_dataset, make_pca_latents, sample_latent_target


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["lfw", "digits", "mnist", "fashionmnist", "cifar10"], default="lfw")
    p.add_argument("--data-root", default="data")
    p.add_argument("--download", action="store_true")
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--k-neighbors", type=int, default=16)
    p.add_argument("--ridge-repeats", type=int, default=8)
    p.add_argument("--ridge", type=float, default=1e-4)
    p.add_argument("--ode-steps", type=int, default=50)
    p.add_argument("--out-dir", default="outputs/realdata_benchmark")
    return p.parse_args()


def _plot_metric(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for metric, marker in [("gap_auc", "o"), ("eval_velocity_mse", "s"), ("latent_mmd2", "^")]:
        vals = df.groupby("coupling")[metric].mean()
        ax.plot(vals.index, vals.values, marker=marker, label=metric)
    ax.set_yscale("log")
    ax.set_title("Real-data latent benchmark")
    ax.set_ylabel("metric value, log scale")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_image_dataset(
        args.dataset,
        data_root=args.data_root,
        train=True,
        download=args.download,
        max_samples=args.max_samples,
        seed=0,
    )
    split = make_pca_latents(dataset, latent_dim=args.latent_dim, seed=0)
    times = np.linspace(0.05, 0.95, 19)
    rows = []

    for seed in args.seeds:
        target_train = sample_latent_target(split.train, args.n, seed=seed + 1000)
        target_eval = sample_latent_target(split.test, args.n, seed=seed + 3000)
        rng = np.random.default_rng(seed + 2000)
        base = rng.normal(size=target_train.shape).astype(np.float64)
        couplings = {
            "independent": independent_pairing(base, target_train, seed=seed),
            "minibatch_ot": minibatch_ot_pairing(base, target_train),
        }
        for name, coupling in couplings.items():
            gap = estimate_markovization_gap(
                coupling.z,
                coupling.x,
                times=times,
                k_neighbors=args.k_neighbors,
            )
            model, train_result = fit_linear_velocity_model(
                coupling.z,
                coupling.x,
                repeats=args.ridge_repeats,
                ridge=args.ridge,
                seed=seed,
            )
            generated = sample_linear_model(
                model,
                n=args.n,
                n_steps=args.ode_steps,
                seed=seed + 4000,
            )
            mmd = rbf_mmd2(generated, target_eval, seed=seed)
            mean_err, cov_err = mean_cov_error(generated, target_eval)
            rows.append(
                {
                    "dataset": args.dataset,
                    "latent_dim": split.train.shape[1],
                    "explained_variance_ratio": split.explained_variance_ratio,
                    "n": args.n,
                    "seed": seed,
                    "coupling": name,
                    "mean_pair_cost": coupling.mean_cost,
                    "gap_auc": gap.auc,
                    "train_mse": train_result.train_mse,
                    "eval_velocity_mse": train_result.eval_mse,
                    "latent_mmd2": mmd,
                    "latent_mean_error": mean_err,
                    "latent_cov_error": cov_err,
                    "ridge_repeats": args.ridge_repeats,
                    "ridge": args.ridge,
                    "ode_steps": args.ode_steps,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "benchmark_results.csv", index=False)
    summary = (
        df.groupby(["dataset", "coupling"])
        .agg(
            gap_auc_mean=("gap_auc", "mean"),
            gap_auc_std=("gap_auc", "std"),
            train_mse_mean=("train_mse", "mean"),
            train_mse_std=("train_mse", "std"),
            eval_velocity_mse_mean=("eval_velocity_mse", "mean"),
            eval_velocity_mse_std=("eval_velocity_mse", "std"),
            latent_mmd2_mean=("latent_mmd2", "mean"),
            latent_mmd2_std=("latent_mmd2", "std"),
            latent_mean_error_mean=("latent_mean_error", "mean"),
            latent_cov_error_mean=("latent_cov_error", "mean"),
            pair_cost_mean=("mean_pair_cost", "mean"),
            explained_variance_ratio=("explained_variance_ratio", "first"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "benchmark_summary.csv", index=False)
    _plot_metric(df, out_dir / "benchmark_metrics.pdf")
    metadata = {
        "dataset": args.dataset,
        "image_shape": dataset.image_shape,
        "num_loaded_samples": int(dataset.x.shape[0]),
        "latent_dim": int(split.train.shape[1]),
        "explained_variance_ratio": split.explained_variance_ratio,
        "seeds": args.seeds,
        "n": args.n,
        "k_neighbors": args.k_neighbors,
        "ridge_repeats": args.ridge_repeats,
        "ridge": args.ridge,
        "ode_steps": args.ode_steps,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(summary.to_string(index=False))
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
