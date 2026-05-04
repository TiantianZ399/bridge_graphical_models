import numpy as np

from psvb.poisson import integrate_field_lines, softened_coulomb_field


def test_softened_coulomb_field_shape_and_finite():
    query = np.array([[0.0, 0.0], [1.0, 0.0]])
    charges = np.array([[0.0, 1.0], [0.0, -1.0]])
    field = softened_coulomb_field(query, charges, softening=0.1)
    assert field.shape == query.shape
    assert np.all(np.isfinite(field))


def test_integrate_field_lines_shape():
    start = np.zeros((3, 2))
    charges = np.array([[1.0, 0.0], [-1.0, 0.0]])

    def field(y):
        return softened_coulomb_field(y, charges, softening=0.1)

    path = integrate_field_lines(start, field, n_steps=4, step_size=0.1)
    assert path.shape == (5, 3, 2)
    assert np.all(np.isfinite(path))
