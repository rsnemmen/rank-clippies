# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Single-file Python CLI (`rank_models.py`) that aggregates LLM benchmark leaderboards into a unified ranking using percentile normalization and statistical tiering. See `AGENTS.md` for full methodology details.

**Repository:** https://github.com/rsnemmen/rank-clippies

## Commands

```bash
# Run rankings  (valid categories: general, coding, agentic, stem)
python rank_models.py general
python rank_models.py coding -p
python rank_models.py general -d -p   # debug + plot
python rank_models.py general -p -q   # plot with quadrant overlays

# Type checking
mypy rank_models.py --strict

# Lint / format
ruff check rank_models.py --fix
ruff format rank_models.py

# Tests / validation
python -m pytest tests/ -v
make validate   # ruff + mypy --strict + pytest + JSON export smoke test
```

## Architecture

Everything lives in `rank_models.py` (stdlib-only for core; pandas/matplotlib/numpy imported lazily inside plotting functions):

- `_load_benchmarks_toml(filepath)` / `_load_models_toml(filepath)` — load TOML data via `tomllib`
- `load_data(category)` — loads `data/benchmarks.toml` + `data/models.toml`, validates benchmark schema, filters by category tag, and returns `(benchmarks, cost_dict, open_dict, title, benchmark_names)`
- `_compute_raw(category)` — computes percentile scores, applies sparse-data penalties, and returns ranked tuples plus raw model scores
- `compute_rankings(category)` — builds the JSON-serializable shape consumed by the static website
- `main()` — handles CLI parsing, ASCII table output, optional plotting, and `--export-json`
- `categorize_tiers()` — groups models into tiers via "Indistinguishable from Best" (asymmetric Q1–Q3 interval overlap); requires pandas
- `create_plot()` — scatter plot (performance vs. cost, log-scale X); cost is normalized so the best-ranked model = 1.0; saves `<basename>.png`; optional `-q`/`--quadrants` flag shades and labels four regions (Best value / Premium / Budget / Avoid) using geometric-mean cost and median score as midpoints
- `create_ranking_plot()` — horizontal ranking chart; saves `<basename>_ranking.png`

## Project Structure

```
.
├── rank_models.py          # CLI, ranking logic, plots, and JSON export
├── data/
│   ├── benchmarks.toml     # Benchmark definitions and scores
│   └── models.toml         # Model cost and open-weight metadata
├── docs/
│   ├── app.js              # Static website logic
│   ├── data/               # Generated JSON consumed by the website
│   └── index.html
├── figures/                # Generated PNG plots (ignored)
└── tests/
    └── test_rank_models.py
```

## Data File Format

Two centralized TOML files in `data/`, parsed with standard-library `tomllib`:

**`data/benchmarks.toml`** — one table per benchmark:
```toml
[benchmark_name]
categories = ["general", "stem"]  # one or more category tags
min_score = 13.2                  # score-based: floor value
# OR known_totals = 347           # rank-based: total models evaluated

[benchmark_name.scores]
model_a = 94.2
# Omit models that were not evaluated on this benchmark.
```
- Score-based (`min_score`): percentile derived from score range; higher = better
- Rank-based (`known_totals`): percentile = rank / total; lower rank number = better
- Omitted scores mean the model was not evaluated on that benchmark
- Valid category tags: `general`, `coding`, `agentic`, `stem` (defined in `CATEGORIES` in `rank_models.py`)

**`data/models.toml`** — one table per model, alphabetical by model name:
```toml
[model_name]
cost = 510    # USD per 1M tokens (input + output)
open = false  # open-weight models get diamond markers in plots

[open_model]
cost = 23
open = true
```
- Missing `cost` means pricing is unknown
- `open = true` gets a diamond marker in scatter plots; `false` = circle

## Code Style

- Python 3.10+; use `dict[str, ...]` and `str | None` union syntax
- 4-space indent, ~100-char line length, double-quoted strings, f-strings for formatting
- All function parameters and return types annotated
- Errors → `sys.exit()` or `print(..., file=sys.stderr)`; info → stdout
- Optional dependency import pattern: `try: import pandas ... except ImportError: sys.exit(...)`
