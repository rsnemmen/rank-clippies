SHELL := /bin/bash

PYTHON ?= python
RANK_SCRIPT := rank_models.py
DOCS_DATA_DIR ?= docs/data
WEBSITE_PLOTS_DIR ?=
CATEGORIES := general coding agentic stem

.PHONY: help refresh-plots refresh-ranking-data check-ranking-data test lint type-check validate

help:
	@echo "Available targets:"
	@echo "  refresh-plots         Generate plots for all categories"
	@echo "  refresh-ranking-data  Regenerate docs/data JSON for the website"
	@echo "  check-ranking-data    Smoke-test JSON export in a temporary directory"
	@echo "  test                  Run pytest"
	@echo "  lint                  Run ruff on rank_models.py"
	@echo "  type-check            Run mypy on rank_models.py"
	@echo "  validate              Run lint, type-check, tests, and JSON export smoke test"
	@echo
	@echo "Optional variables:"
	@echo "  PYTHON=<python>                 Python executable to use"
	@echo "  DOCS_DATA_DIR=<dir>             Output directory for refresh-ranking-data"
	@echo "  WEBSITE_PLOTS_DIR=<dir>         Optional destination for copied PNGs"

refresh-plots:
	@set -euo pipefail; \
	for category in $(CATEGORIES); do \
		printf '\n[%s]\n' "$$category"; \
		$(PYTHON) $(RANK_SCRIPT) "$$category" --plot --quadrants; \
	done; \
	if [[ -n "$(WEBSITE_PLOTS_DIR)" ]]; then \
		mkdir -p "$(WEBSITE_PLOTS_DIR)"; \
		cp figures/*.png "$(WEBSITE_PLOTS_DIR)/"; \
		echo "Copied plots to $(WEBSITE_PLOTS_DIR)"; \
	fi

refresh-ranking-data:
	$(PYTHON) $(RANK_SCRIPT) --export-json "$(DOCS_DATA_DIR)"

check-ranking-data:
	@set -euo pipefail; \
	tmp_dir="$$(mktemp -d)"; \
	trap 'rm -rf "$$tmp_dir"' EXIT; \
	$(PYTHON) $(RANK_SCRIPT) --export-json "$$tmp_dir" >/dev/null

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	ruff check $(RANK_SCRIPT)

type-check:
	mypy $(RANK_SCRIPT) --strict

validate: lint type-check test check-ranking-data
