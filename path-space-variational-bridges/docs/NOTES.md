# Notes

## JMLR + natbib

The paper source uses the JMLR class. Do not reload numeric `natbib` inside `paper/main.tex`.
The previous failing source mixed a JMLR template with `\usepackage[numbers,sort&compress]{natbib}`, which triggers a `natbib` incompatibility.

## Markovization-gap diagnostic

The toy experiment estimates

```math
\mathfrak G(t) = \mathbb E\,\mathrm{Var}(U_t \mid X_t)
```

using a k-nearest-neighbor conditional mean approximation. The estimator in `src/psvb/gap.py` is intended as a relative bridge/coupling diagnostic, not as a high-accuracy nonparametric estimator.

## Field-line bridge utilities

`src/psvb/poisson.py` contains a softened point-charge field and Euler integration. This demonstrates the field-line-bridge idea at toy scale. It is not a replacement for PFGM, PFGM++, EFM, IFM, or flow/field matching training code.
