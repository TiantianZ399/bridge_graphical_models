"""Tiny example showing the package API."""

import numpy as np

from psvb import (
    estimate_markovization_gap,
    minibatch_ot_pairing,
    sample_eight_gaussians,
    sample_standard_normal,
)

z = sample_standard_normal(128, dim=2, seed=0)
x = sample_eight_gaussians(128, seed=1)
coupling = minibatch_ot_pairing(z, x)
result = estimate_markovization_gap(coupling.z, coupling.x, np.linspace(0.1, 0.9, 5), k_neighbors=16)
print(result)
