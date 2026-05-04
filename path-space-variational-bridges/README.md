# Path-Space Variational Bridges

Code and paper source for **Path-Space Variational Bridges: Couplings, Gauges, and Field-Line Bridges for Generative Modeling**.

This repository is set up as a GitHub-ready companion to the preliminary manuscript. It contains:

- a JMLR-style LaTeX paper source in `paper/`,
- a small Python package `psvb`,
- reproducible 2D diagnostics for the Markovization gap,
- toy electrostatic/Poisson field-line utilities,
- tests and GitHub Actions workflows.

The main diagnostic estimates the **Markovization gap** for a straight-line bridge under two endpoint couplings: independent pairing and minibatch optimal-transport pairing.

## Repository layout

```text
.
├── paper/                         # JMLR-style LaTeX source and compiled PDF
│   ├── main.tex
│   ├── references.bib
│   ├── main.bbl
│   ├── figures/
│   └── path_space_variational_bridges_jmlr_v2.pdf
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

## Optional field-line demo

```bash
python examples/field_line_demo.py
```

This demo makes the Poisson/electrostatic bridge idea concrete, but it is not a full PFGM implementation.

```

## Citation

See [`CITATION.cff`](CITATION.cff). The manuscript is preliminary; update the citation metadata when there is an arXiv identifier or camera-ready version.
