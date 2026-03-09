# ── cluster_pipeline convenience targets ──────────────────────────
SHELL   := /bin/bash
VENV    := .venv
PYTHON  := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
PYTEST  := $(VENV)/bin/pytest

.PHONY: setup setup-quick test lint ci clean help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## Full install: venv + pip + SExtractor + BAOlab
	bash scripts/setup_env.sh

setup-quick:  ## Quick install: venv + pip only (no ext tools)
	bash scripts/setup_env.sh --quick

test:  ## Run pytest
	$(PYTEST) tests/ -v

lint:  ## Run flake8 + mypy
	$(VENV)/bin/flake8 cluster_pipeline/ tests/
	$(VENV)/bin/mypy cluster_pipeline/ --ignore-missing-imports

ci: lint test  ## Run full CI (lint + test)

clean:  ## Remove venv and build artefacts
	rm -rf $(VENV) .deps/ __pycache__ .mypy_cache .pytest_cache
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
