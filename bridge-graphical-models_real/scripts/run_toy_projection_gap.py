#!/usr/bin/env python
"""Reproduce the Markovization-gap toy diagnostic from the paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from psvb import (
    estimate_markovization_gap,
    independent_pairing,
    minibatch_ot_pairing,
    sample_eight_gaussians,
    sample_standard_normal,
)
from psvb.plotting import plot_toy_projection_gap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000, help="number of source/target samples")
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument("--radius", type=float, default=3.0, help="eight-Gaussian radius")
    parser.add_argument("--noise", type=float, default=0.25, help="eight-Gaussian component noise")
    parser.add_argument("--k-neighbors", type=int, default=32, help="k for k-NN conditional mean")
    parser.add_argument("--num-times", type=int, default=19, help="number of bridge times")
    parser.add_argument(
        "--out-dir",
        "--output-dir",
        dest="out_dir",
        type=Path,
        default=Path("outputs/toy_projection_gap"),
        help="directory for CSV and figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    z = sample_standard_normal(args.n, dim=2, seed=args.seed)
    x_raw = sample_eight_gaussians(args.n, radius=args.radius, noise=args.noise, seed=args.seed + 1)

    independent = independent_pairing(z, x_raw, seed=args.seed + 2, shuffle=True)
    ot = minibatch_ot_pairing(z, x_raw, power=2.0)

    times = np.linspace(0.05, 0.95, args.num_times)
    gap_ind = estimate_markovization_gap(
        independent.z, independent.x, times, k_neighbors=args.k_neighbors
    )
    gap_ot = estimate_markovization_gap(ot.z, ot.x, times, k_neighbors=args.k_neighbors)

    csv = args.out_dir / "toy_projection_gap.csv"
    out = np.column_stack([times, gap_ind.gaps, gap_ot.gaps])
    np.savetxt(csv, out, delimiter=",", header="t,independent_gap,ot_gap", comments="")

    pdf, png = plot_toy_projection_gap(
        z=z,
        x_target=x_raw,
        times=times,
        independent_gaps=gap_ind.gaps,
        ot_gaps=gap_ot.gaps,
        independent_auc=gap_ind.auc,
        ot_auc=gap_ot.auc,
        out_dir=args.out_dir,
    )

    print(f"independent mean endpoint cost: {independent.mean_cost:.6f}")
    print(f"minibatch OT mean endpoint cost: {ot.mean_cost:.6f}")
    print(f"independent Markovization-gap AUC: {gap_ind.auc:.6f}")
    print(f"minibatch OT Markovization-gap AUC: {gap_ot.auc:.6f}")
    print(f"wrote {csv}")
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
