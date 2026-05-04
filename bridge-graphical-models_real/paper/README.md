# Bridge Graphical Models -- NeurIPS main-track draft

This folder contains an anonymous NeurIPS-main-track style draft for the paper
formerly titled "Path-Space Variational Bridges".

## Files

- `main.tex`: anonymous NeurIPS-style manuscript.
- `neurips_2026.sty`: local-compatible style file used for compilation here. Before final submission, replace this with the official `neurips_2026.sty` from neurips.cc / the official Overleaf template.
- `references.bib`: bibliography.
- `checklist.tex`: NeurIPS paper checklist.
- `figures/`: generated toy diagnostic figures and CSVs.
- `scripts/toy_projection_gap_replicates.py`: reproducibility script for the replication diagnostic.

## Compile

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Reproduce toy diagnostic

```bash
python scripts/toy_projection_gap_replicates.py --out-dir figures
```

Dependencies: NumPy, SciPy, scikit-learn, Matplotlib.

## Notes

The PDF is anonymous by default. For arXiv, use the official NeurIPS preprint option and restore the author block. The main text is designed to fit within the 9-page NeurIPS content limit; references, appendices, and checklist follow afterward.
