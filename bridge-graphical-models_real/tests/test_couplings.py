import numpy as np

from psvb import independent_pairing, minibatch_ot_pairing


def test_independent_pairing_is_permutation():
    z = np.zeros((5, 2))
    x = np.arange(10, dtype=float).reshape(5, 2)
    result = independent_pairing(z, x, seed=0, shuffle=True)
    assert result.x.shape == x.shape
    assert sorted(map(tuple, result.x)) == sorted(map(tuple, x))
    assert result.mean_cost >= 0.0


def test_minibatch_ot_pairing_is_sorted_for_one_dimensional_case():
    z = np.array([[0.0], [2.0], [4.0]])
    x = np.array([[4.1], [0.1], [2.1]])
    result = minibatch_ot_pairing(z, x)
    assert result.x.shape == x.shape
    assert sorted(map(tuple, result.x)) == sorted(map(tuple, x))
    assert np.allclose(result.x[:, 0], [0.1, 2.1, 4.1])
