# Bridge Graphical Models

Code and paper source for **Bridge Graphical Models: Coupling, Projection, and Gauge in Generative Modeling**.

This repository is set up as a GitHub-ready companion to the preliminary manuscript. It contains:

- a NeurIPS-style LaTeX paper source in `paper/`,
- a small Python package `psvb`,
- reproducible 2D diagnostics for the Markovization gap,
- real-data latent benchmarks on scikit-image LFW subset and torchvision MNIST/Fashion-MNIST/CIFAR-10,
- toy electrostatic/Poisson field-line utilities,
- tests and GitHub Actions workflows.

The main diagnostic estimates the **Markovization gap** for a straight-line bridge under two endpoint couplings: independent pairing and minibatch optimal-transport pairing.

## Repository layout

```text
.
├── paper/                         # NeurIPS-style LaTeX source and compiled PDF
│   ├── main.tex
│   ├── references.bib
│   ├── main.bbl
│   ├── figures/
│   └── main.pdf
├── src/psvb/                      # Python package
│   ├── bridges.py                 # straight and Brownian bridge primitives
│   ├── couplings.py               # independent and minibatch-OT endpoint pairings
│   ├── distributions.py           # toy source/target distributions
│   ├── gap.py                     # k-NN Markovization-gap estimator
│   ├── plotting.py                # figure helpers
│   └── poisson.py                 # toy field-line bridge utilities
├── scripts/
│   ├── run_toy_projection_gap.py  # reproduces paper/figures/toy_projection_gap.*
│   └── compile_paper.sh
├── examples/
│   ├── demo_import.py
│   └── field_line_demo.py
├── tests/
├── docs/
├── pyproject.toml
├── requirements.txt
├── environment.yml
├── Makefile
├── CITATION.cff
└── LICENSE
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Alternative without editable install:

```bash
pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

Run tests:

```bash
pytest
```

## Reproduce the toy Markovization-gap figure

```bash
python scripts/run_toy_projection_gap.py --n 1000 --seed 7 --output-dir paper/figures
```

or:

```bash
make toy
```

Expected qualitative result: minibatch OT pairing should give a much smaller integrated Markovization gap than independent pairing on the eight-Gaussian toy target.

## Minimal package API

```python
import numpy as np
from psvb import (
    sample_standard_normal,
    sample_eight_gaussians,
    minibatch_ot_pairing,
    estimate_markovization_gap,
)

z = sample_standard_normal(512, dim=2, seed=0)
x = sample_eight_gaussians(512, seed=1)
coupling = minibatch_ot_pairing(z, x)
result = estimate_markovization_gap(coupling.z, coupling.x, np.linspace(0.1, 0.9, 9))
print(result.auc)
```


## Real-data latent benchmarks

The repo now includes a real-data benchmark suite. The default `lfw` benchmark is no-download; MNIST, Fashion-MNIST, and CIFAR-10 use torchvision and can be downloaded locally.

Gap-only benchmark:

```bash
PYTHONPATH=src python scripts/run_realdata_gap.py --dataset lfw --n 150 --seeds 0 1 2
PYTHONPATH=src python scripts/run_realdata_gap.py --dataset mnist --download --n 2000 --seeds 0 1 2
PYTHONPATH=src python scripts/run_realdata_gap.py --dataset fashionmnist --download --n 2000 --seeds 0 1 2
PYTHONPATH=src python scripts/run_realdata_gap.py --dataset cifar10 --download --latent-dim 32 --n 2000 --seeds 0 1 2
```

Train-and-sample latent benchmark:

```bash
PYTHONPATH=src python scripts/run_realdata_benchmark.py --dataset lfw --n 150 --ridge-repeats 8 --seeds 0 1 2
```

This benchmark does not chase state-of-the-art FID. It tests the paper's specific prediction: lower Markovization gap should correlate with lower velocity-regression error and lower latent distribution mismatch under fixed architecture and compute.

See `benchmarks/README.md` for the full protocol.

## Current LFW result

The NeurIPS-style draft includes the following no-download real-image latent benchmark, generated from the scikit-image LFW subset with 16-dimensional PCA-whitened latents over seeds 0, 1, and 2.

| Coupling | Gap AUC ↓ | Eval velocity MSE ↓ | Latent MMD² ↓ |
|---|---:|---:|---:|
| independent | 23.701 ± 0.880 | 23.511 ± 1.385 | 0.02780 ± 0.00649 |
| minibatch OT | 11.871 ± 0.201 | 12.877 ± 0.078 | 0.02744 ± 0.00686 |

The result supports the gap/training-difficulty claim: OT roughly halves the Markovization gap and reduces fixed-ridge velocity error under the same bridge. It does not claim state-of-the-art image generation.

## Optional field-line demo

```bash
python examples/field_line_demo.py
```

This demo makes the Poisson/electrostatic bridge idea concrete, but it is not a full PFGM implementation.

## Compile the paper

The JMLR class already manages `natbib`; do **not** add numeric `natbib` options to `paper/main.tex`.

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Since `paper/main.bbl` is included, this often works with only:

```bash
cd paper
pdflatex main.tex
pdflatex main.tex
```

From the repository root:

```bash
make paper
```

## Push to GitHub

```bash
git init
git add .
git commit -m "Initial path-space variational bridges repo"
git branch -M main
git remote add origin git@github.com:<your-username>/path-space-variational-bridges.git
git push -u origin main
```

## Citation

See [`CITATION.cff`](CITATION.cff). The manuscript is preliminary; update the citation metadata when there is an arXiv identifier or camera-ready version.

The `paper/` folder includes the full NeurIPS checklist with section-referenced answers and an ethics/assets/compute appendix.
