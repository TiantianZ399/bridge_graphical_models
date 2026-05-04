.PHONY: install test toy example paper clean

install:
	pip install -e '.[dev]'

test:
	pytest

toy:
	python scripts/run_toy_projection_gap.py --n 1000 --seed 7 --output-dir paper/figures

example:
	python examples/demo_import.py
	python examples/field_line_demo.py

paper:
	bash scripts/compile_paper.sh

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find paper -maxdepth 1 \( -name '*.aux' -o -name '*.log' -o -name '*.out' -o -name '*.blg' -o -name '*.fls' -o -name '*.fdb_latexmk' -o -name '*.synctex.gz' -o -name '*.toc' \) -delete
