import numpy as np

from psvb import (
    brownian_bridge_marginal,
    estimate_markovization_gap,
    independent_pairing,
    minibatch_ot_pairing,
    sample_eight_gaussians,
    sample_standard_normal,
    straight_interpolation,
    straight_velocity,
)
from psvb.poisson import integrate_field_lines, softened_coulomb_field


def test_straight_bridge_endpoints_and_velocity():
    z = np.zeros((4, 2))
    x = np.ones((4, 2))
    np.testing.assert_allclose(straight_interpolation(z, x, 0.0), z)
    np.testing.assert_allclose(straight_interpolation(z, x, 1.0), x)
    np.testing.assert_allclose(straight_velocity(z, x), np.ones((4, 2)))


def test_ot_pairing_has_no_larger_cost_than_random_for_small_batch():
    z = sample_standard_normal(64, seed=0)
    x = sample_eight_gaussians(64, seed=1)
    independent = independent_pairing(z, x, seed=2)
    ot = minibatch_ot_pairing(z, x)
    assert ot.mean_cost <= independent.mean_cost


def test_markovization_gap_is_finite():
    z = sample_standard_normal(128, seed=0)
    x = sample_eight_gaussians(128, seed=1)
    result = estimate_markovization_gap(z, x, np.linspace(0.1, 0.9, 5), k_neighbors=16)
    assert result.gaps.shape == (5,)
    assert np.all(np.isfinite(result.gaps))
    assert np.isfinite(result.auc)


def test_constant_velocity_gap_is_numerically_zero():
    z = sample_standard_normal(64, seed=0)
    x = z + np.array([1.0, -0.5])
    result = estimate_markovization_gap(z, x, np.linspace(0.1, 0.9, 5), k_neighbors=8)
    assert result.auc < 1e-12


def test_brownian_bridge_marginal_shape():
    z = np.zeros((10, 2))
    x = np.ones((10, 2))
    sample = brownian_bridge_marginal(z, x, t=0.5, epsilon=0.1, seed=0)
    assert sample.shape == z.shape


def test_softened_field_and_field_lines_shape():
    charges = np.array([[0.0, 0.0], [1.0, 0.0]])
    query = np.array([[0.5, 1.0], [0.5, 2.0]])
    field = softened_coulomb_field(query, charges)
    assert field.shape == query.shape
    path = integrate_field_lines(query, lambda y: softened_coulomb_field(y, charges), n_steps=3)
    assert path.shape == (4, 2, 2)
