#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../paper"
pdflatex -interaction=nonstopmode -halt-on-error main.tex
if command -v bibtex >/dev/null 2>&1; then
  bibtex main
else
  echo "bibtex not found; using included main.bbl"
fi
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
