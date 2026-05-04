#!/usr/bin/env python3
"""Tiny field-line demo for the Poisson/electrostatic bridge utilities."""

from __future__ import annotations

import numpy as np

from psvb.distributions import sample_eight_gaussians
from psvb.poisson import integrate_field_lines, softened_coulomb_field


def main() -> None:
    charges = np.pad(sample_eight_gaussians(64, seed=0), ((0, 0), (0, 1)))
    y0 = np.array([[0.0, 0.0, 5.0], [2.0, 0.0, 5.0], [-2.0, 1.0, 5.0]])

    def field(y: np.ndarray) -> np.ndarray:
        return softened_coulomb_field(y, charges, softening=0.2)

    path = integrate_field_lines(y0, field, step_size=0.02, n_steps=100, direction=-1.0)
    print("path shape:", path.shape)
    print("final states:\n", path[-1])


if __name__ == "__main__":
    main()
