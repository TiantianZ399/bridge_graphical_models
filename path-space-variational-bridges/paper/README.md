# Paper source

This directory contains the JMLR-style LaTeX source for the preliminary manuscript.

## Compile

The JMLR class already handles `natbib`. Do **not** add `\usepackage[numbers]{natbib}`.

Because `main.bbl` is included, compile with:

```bash
pdflatex main.tex
pdflatex main.tex
```

To regenerate the bibliography after editing `references.bib`:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

From the repository root, you can also run:

```bash
bash scripts/compile_paper.sh
```

## Toy figure

The figure in `figures/toy_projection_gap.pdf` can be regenerated from the repository root:

```bash
pip install -e .
python scripts/run_toy_projection_gap.py --out-dir paper/figures
```
