import numpy as np

from psvb.metrics import mean_cov_error, rbf_mmd2
from psvb.realdata import load_image_dataset, make_pca_latents, sample_latent_target


def test_lfw_pca_latents():
    ds = load_image_dataset("lfw", max_samples=150, seed=0)
    split = make_pca_latents(ds, latent_dim=8, seed=0)
    assert split.train.shape[1] == 8
    assert split.test.shape[1] == 8
    assert 0.0 < split.explained_variance_ratio <= 1.0
    sample = sample_latent_target(split.train, 32, seed=1)
    assert sample.shape == (32, 8)


def test_latent_metrics_are_finite():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 4))
    y = rng.normal(size=(64, 4))
    mmd = rbf_mmd2(x, y)
    mean_err, cov_err = mean_cov_error(x, y)
    assert np.isfinite(mmd)
    assert np.isfinite(mean_err)
    assert np.isfinite(cov_err)
    assert mmd >= -1e-8
