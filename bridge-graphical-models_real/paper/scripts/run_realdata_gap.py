#!/usr/bin/env python
"""Estimate Markovization gaps on real datasets in PCA latent space."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from psvb.couplings import independent_pairing, minibatch_ot_pairing
from psvb.gap import estimate_markovization_gap
from psvb.realdata import load_image_dataset, make_pca_latents, sample_latent_target


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["lfw", "digits", "mnist", "fashionmnist", "cifar10"], default="lfw")
    p.add_argument("--data-root", default="data")
    p.add_argument("--download", action="store_true")
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--k-neighbors", type=int, default=32)
    p.add_argument("--out-dir", default="outputs/realdata_gap")
    return p.parse_args()


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
        target = sample_latent_target(split.train, args.n, seed=seed + 1000)
        rng = np.random.default_rng(seed + 2000)
        base = rng.normal(size=target.shape).astype(np.float64)
        for coupling_name, coupling in [
            ("independent", independent_pairing(base, target, seed=seed)),
            ("minibatch_ot", minibatch_ot_pairing(base, target)),
        ]:
            result = estimate_markovization_gap(
                coupling.z,
                coupling.x,
                times=times,
                k_neighbors=args.k_neighbors,
            )
            rows.append(
                {
                    "dataset": args.dataset,
                    "latent_dim": split.train.shape[1],
                    "explained_variance_ratio": split.explained_variance_ratio,
                    "n": args.n,
                    "seed": seed,
                    "coupling": coupling_name,
                    "mean_pair_cost": coupling.mean_cost,
                    "gap_auc": result.auc,
                    "k_neighbors": args.k_neighbors,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "gap_results.csv", index=False)
    summary = (
        df.groupby(["dataset", "coupling"])
        .agg(
            gap_auc_mean=("gap_auc", "mean"),
            gap_auc_std=("gap_auc", "std"),
            pair_cost_mean=("mean_pair_cost", "mean"),
            explained_variance_ratio=("explained_variance_ratio", "first"),
            n=("n", "first"),
            latent_dim=("latent_dim", "first"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "gap_summary.csv", index=False)
    metadata = {
        "dataset": args.dataset,
        "image_shape": dataset.image_shape,
        "num_loaded_samples": int(dataset.x.shape[0]),
        "latent_dim": int(split.train.shape[1]),
        "explained_variance_ratio": split.explained_variance_ratio,
        "seeds": args.seeds,
        "n": args.n,
        "k_neighbors": args.k_neighbors,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(summary.to_string(index=False))
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
