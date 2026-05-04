"""Utilities for Path-Space Variational Bridges.

The package is intentionally small and focused on the preliminary paper's
reproducible diagnostics: endpoint couplings, bridge primitives, k-NN estimates
of the Markovization gap, and simple field-line helpers.
"""

from .bridges import brownian_bridge_marginal, straight_interpolation, straight_velocity
from .couplings import CouplingResult, independent_pairing, minibatch_ot_pairing
from .distributions import sample_eight_gaussians, sample_standard_normal
from .gap import MarkovizationGapResult, estimate_markovization_gap
from .realdata import ImageDataset, LatentSplit, load_image_dataset, make_pca_latents, sample_latent_target
from .metrics import rbf_mmd2, mean_cov_error

__all__ = [
    "brownian_bridge_marginal",
    "straight_interpolation",
    "straight_velocity",
    "CouplingResult",
    "independent_pairing",
    "minibatch_ot_pairing",
    "sample_eight_gaussians",
    "sample_standard_normal",
    "MarkovizationGapResult",
    "estimate_markovization_gap",
    "ImageDataset",
    "LatentSplit",
    "load_image_dataset",
    "make_pca_latents",
    "sample_latent_target",
    "rbf_mmd2",
    "mean_cov_error",
]
