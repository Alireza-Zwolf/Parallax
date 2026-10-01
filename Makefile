.PHONY: help install test archive clean

help:
	@echo "Targets: install test archive clean"

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest -q

# Uses archived responses, embeddings, and judge scores. No provider calls or model downloads.
archive:
	parallax-audit archive-run --source datasets --outputs outputs

clean:
	rm -rf outputs build src/parallax_audit.egg-info .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
