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
```

No tests exist yet; if added, place them in `tests/` and run with `pytest tests/ -v`.

## Architecture

Everything lives in `rank_models.py` (stdlib-only for core; pandas/matplotlib/numpy imported lazily inside plotting functions):

- `load_data(category)` — loads `data/benchmarks.txt` + `data/models.txt`, filters by category tag, returns `(benchmarks, cost_dict, open_dict, title)`
- `_extract_raw_dicts(filepath)` — brace-depth parser that extracts top-level dict literals from a file
- `main()` — computes percentile scores, applies sparse-data penalty, prints ASCII table, optionally calls plotting functions
- `categorize_tiers()` — groups models into tiers via "Indistinguishable from Best" (±semi-IQR interval overlap); requires pandas
- `create_plot()` — scatter plot (performance vs. cost, log-scale X); cost is normalized so the best-ranked model = 1.0; saves `<basename>.png`; optional `-q`/`--quadrants` flag shades and labels four regions (Best value / Premium / Budget / Avoid) using geometric-mean cost and median score as midpoints
- `create_ranking_plot()` — horizontal ranking chart; saves `<basename>_ranking.png`

## Data File Format

Two centralized files in `data/`, both containing Python dict literals parsed with `ast.literal_eval`:

**`data/benchmarks.txt`** — one dict per benchmark, alphabetical order:
```python
benchmark_name = {
    "categories": ["general", "stem"],   # one or more category tags
    "min_score": 13.2,                   # score-based: floor value
    # OR "known_totals": 347,            # rank-based: total models evaluated
    "scores": {"model_a": 94.2, "model_b": None, ...},  # alphabetical by model
}
```
- Score-based (`min_score`): percentile derived from score range; higher = better
- Rank-based (`known_totals`): percentile = rank / total; lower rank number = better
- `None` scores mean the model was not evaluated on that benchmark
- Valid category tags: `general`, `coding`, `agentic`, `stem` (defined in `CATEGORIES` in `rank_models.py`)

**`data/models.txt`** — single `models` dict, alphabetical by model name:
```python
models = {
    "model_name": {"cost": 510, "open": False},  # cost = credits per 1k tokens (Poe)
    "open_model":  {"cost": 23,  "open": True},  # open-weight: gets diamond marker in plots
}
```
- `cost: None` — model is tracked but pricing is unknown
- `open: True` — model gets a diamond marker in scatter plots; `False` = circle

Full-line `#` comments are stripped before parsing.

## Code Style

- Python 3.10+; use `dict[str, ...]` and `str | None` union syntax
- 4-space indent, ~100-char line length, double-quoted strings, f-strings for formatting
- All function parameters and return types annotated
- Errors → `sys.exit()` or `print(..., file=sys.stderr)`; info → stdout
- Optional dependency import pattern: `try: import pandas ... except ImportError: sys.exit(...)`
