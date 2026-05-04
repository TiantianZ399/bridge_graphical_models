"""Real-data latent benchmarks for Bridge Graphical Models.

This module builds low-cost latent benchmarks from real image datasets. It uses a
simple PCA-whitened representation so the benchmark isolates the BGM question:

    coupling/bridge choice -> Markovization gap -> training and sample behavior.

The no-download default is ``lfw`` from scikit-image. Standard datasets are
available through torchvision when run locally with ``--download``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
DatasetName = Literal["lfw", "digits", "mnist", "fashionmnist", "cifar10"]


@dataclass(frozen=True)
class ImageDataset:
    """Flattened real-image dataset."""

    x: Array
    y: NDArray[np.int64]
    image_shape: tuple[int, ...]
    name: str


@dataclass(frozen=True)
class PCATransformer:
    """A minimal PCA-whitening transformer implemented with NumPy."""

    mean: Array
    scale: Array
    components: Array
    explained_variance: Array
    explained_variance_ratio: Array
    whiten: bool = True

    def transform(self, x: Array) -> Array:
        xs = (np.asarray(x, dtype=np.float64) - self.mean) / self.scale
        z = xs @ self.components.T
        if self.whiten:
            z = z / np.sqrt(self.explained_variance + 1e-12)
        return z.astype(np.float64)


@dataclass(frozen=True)
class LatentSplit:
    """Train/test latent features with fitted preprocessing objects."""

    train: Array
    test: Array
    y_train: NDArray[np.int64]
    y_test: NDArray[np.int64]
    pca: PCATransformer
    explained_variance_ratio: float
    dataset: str


def load_image_dataset(
    name: DatasetName,
    data_root: str | Path = "data",
    train: bool = True,
    download: bool = False,
    max_samples: int | None = None,
    seed: int = 0,
) -> ImageDataset:
    """Load a real image dataset and return flattened pixels in [0, 1].

    ``lfw`` is a no-download smoke benchmark using scikit-image's included LFW
    subset. ``mnist``, ``fashionmnist`` and ``cifar10`` use torchvision. ``digits``
    uses scikit-learn if available, but ``lfw`` is preferred for no-download runs
    in minimal environments.
    """

    rng = np.random.default_rng(seed)
    name_l = name.lower()

    if name_l == "lfw":
        from skimage import data

        images = data.lfw_subset().astype(np.float64)
        images = np.clip(images, 0.0, 1.0)
        labels = np.arange(images.shape[0], dtype=np.int64)  # identity index, not used by metrics
        flat = images.reshape(images.shape[0], -1)
        image_shape = tuple(images.shape[1:])
    elif name_l == "digits":
        from sklearn.datasets import load_digits  # optional; may be slow in some minimal Python builds

        data_digits = load_digits()
        images = data_digits.images.astype(np.float64) / 16.0
        labels = data_digits.target.astype(np.int64)
        flat = images.reshape(images.shape[0], -1)
        image_shape = tuple(images.shape[1:])
    elif name_l in {"mnist", "fashionmnist", "cifar10"}:
        try:
            import torchvision.datasets as dsets
            import torchvision.transforms as T
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("torchvision is required for MNIST/FashionMNIST/CIFAR10") from exc

        root = Path(data_root)
        transform = T.ToTensor()
        if name_l == "mnist":
            ds = dsets.MNIST(root=str(root), train=train, download=download, transform=transform)
        elif name_l == "fashionmnist":
            ds = dsets.FashionMNIST(root=str(root), train=train, download=download, transform=transform)
        else:
            ds = dsets.CIFAR10(root=str(root), train=train, download=download, transform=transform)

        indices = np.arange(len(ds))
        if max_samples is not None and max_samples < len(ds):
            indices = rng.choice(indices, size=max_samples, replace=False)
        xs: list[np.ndarray] = []
        ys: list[int] = []
        for i in indices:
            img, label = ds[int(i)]
            arr = img.numpy().astype(np.float64)
            xs.append(arr.reshape(-1))
            ys.append(int(label))
        flat = np.stack(xs, axis=0)
        labels = np.asarray(ys, dtype=np.int64)
        image_shape = tuple(ds[0][0].shape)
    else:
        raise ValueError(f"unknown dataset: {name}")

    if max_samples is not None and name_l in {"lfw", "digits"} and max_samples < flat.shape[0]:
        idx = rng.choice(flat.shape[0], size=max_samples, replace=False)
        flat = flat[idx]
        labels = labels[idx]

    return ImageDataset(x=flat.astype(np.float64), y=labels, image_shape=image_shape, name=name_l)


def _random_split(n: int, test_size: float, seed: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = max(1, int(round(test_size * n)))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    if train_idx.size == 0:
        raise ValueError("test_size leaves no training samples")
    return train_idx.astype(np.int64), test_idx.astype(np.int64)


def _fit_pca(x_train: Array, latent_dim: int, whiten: bool = True) -> PCATransformer:
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    xs = (x_train - mean) / scale
    n = xs.shape[0]
    n_components = min(latent_dim, xs.shape[0], xs.shape[1])
    # Economy SVD: xs = U S Vt. Principal axes are rows of Vt.
    # Limit BLAS threads to avoid oversubscription/hangs in small CI containers.
    try:
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1):
            _u, s, vt = np.linalg.svd(xs, full_matrices=False)
    except Exception:  # pragma: no cover - fallback when threadpoolctl is absent
        _u, s, vt = np.linalg.svd(xs, full_matrices=False)
    components = vt[:n_components]
    explained_variance = (s[:n_components] ** 2) / max(n - 1, 1)
    total_variance = np.sum((s**2) / max(n - 1, 1))
    explained_variance_ratio = explained_variance / max(total_variance, 1e-12)
    return PCATransformer(
        mean=mean.astype(np.float64),
        scale=scale.astype(np.float64),
        components=components.astype(np.float64),
        explained_variance=explained_variance.astype(np.float64),
        explained_variance_ratio=explained_variance_ratio.astype(np.float64),
        whiten=whiten,
    )


def make_pca_latents(
    dataset: ImageDataset,
    latent_dim: int = 16,
    test_size: float = 0.25,
    seed: int = 0,
    whiten: bool = True,
) -> LatentSplit:
    """Standardize pixels and build PCA-whitened latent features."""

    if latent_dim <= 0:
        raise ValueError("latent_dim must be positive")
    train_idx, test_idx = _random_split(dataset.x.shape[0], test_size, seed)
    x_train = dataset.x[train_idx]
    x_test = dataset.x[test_idx]
    y_train = dataset.y[train_idx]
    y_test = dataset.y[test_idx]
    pca = _fit_pca(x_train, latent_dim=latent_dim, whiten=whiten)
    z_train = pca.transform(x_train)
    z_test = pca.transform(x_test)
    evr = float(np.sum(pca.explained_variance_ratio))
    return LatentSplit(
        train=z_train,
        test=z_test,
        y_train=y_train.astype(np.int64),
        y_test=y_test.astype(np.int64),
        pca=pca,
        explained_variance_ratio=evr,
        dataset=dataset.name,
    )


def sample_latent_target(latents: Array, n: int, seed: int = 0) -> Array:
    """Sample target latent points with replacement."""

    if n <= 0:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, latents.shape[0], size=n)
    return np.asarray(latents[idx], dtype=np.float64)
