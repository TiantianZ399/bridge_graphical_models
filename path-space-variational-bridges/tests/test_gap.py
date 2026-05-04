import numpy as np

from psvb import estimate_markovization_gap


def test_knn_gap_shape_nonnegative():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(64, 2))
    x = z + 0.1 * rng.normal(size=(64, 2))
    times = np.linspace(0.1, 0.9, 5)
    result = estimate_markovization_gap(z, x, times, k_neighbors=8)
    assert result.gaps.shape == times.shape
    assert np.all(result.gaps >= -1e-12)
    assert np.isfinite(result.auc)


def test_constant_velocity_has_zero_gap():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(128, 2))
    x = z + np.array([1.0, -0.5])
    times = np.linspace(0.1, 0.9, 5)
    result = estimate_markovization_gap(z, x, times, k_neighbors=16)
    assert result.auc < 1e-12
