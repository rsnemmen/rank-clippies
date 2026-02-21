# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Single-file Python CLI (`rank_models.py`) that aggregates LLM benchmark leaderboards into a unified ranking using percentile normalization and statistical tiering. See `AGENTS.md` for full methodology details.

**Repository:** https://github.com/rsnemmen/rank-clippies

## Commands

```bash
# Run rankings
python rank_models.py data/ranks_general.txt
python rank_models.py data/ranks_coding.txt --plot
python rank_models.py data/ranks_general.txt -d -p   # debug + plot

# Type checking
mypy rank_models.py --strict

# Lint / format
ruff check rank_models.py --fix
ruff format rank_models.py
```

No tests exist yet; if added, place them in `tests/` and run with `pytest tests/ -v`.

## Architecture

Everything lives in `rank_models.py` (stdlib-only for core; pandas/matplotlib/numpy imported lazily inside plotting functions):

- `parse_file()` — reads a `.txt` data file containing Python dict literals; returns `(benchmarks, cost_dict, open_dict)`
- `main()` — computes percentile scores, applies sparse-data penalty, prints ASCII table, optionally calls plotting functions
- `categorize_tiers()` — groups models into tiers via "Indistinguishable from Best" (1σ CI overlap); requires pandas
- `create_plot()` — scatter plot (performance vs. cost, log-scale X); saves `<basename>.png`
- `create_ranking_plot()` — horizontal ranking chart; saves `<basename>_ranking.png`

## Data File Format

Files in `data/` contain Python dict literals parsed with `ast.literal_eval`:
- **Benchmark dicts** — `name={model: rank, ..., "known_totals": N}` (one per leaderboard)
- **Open-source dict** (optional) — `{model: True, ...}` (no `known_totals`, no `cost` key); controls marker shape in plots
- **Cost dict** (last) — `{model: credits_per_1k_tokens, ...}` (no `known_totals`)

Full-line `#` comments are stripped before parsing.

## Code Style

- Python 3.10+; use `dict[str, ...]` and `str | None` union syntax
- 4-space indent, ~100-char line length, double-quoted strings, f-strings for formatting
- All function parameters and return types annotated
- Errors → `sys.exit()` or `print(..., file=sys.stderr)`; info → stdout
- Optional dependency import pattern: `try: import pandas ... except ImportError: sys.exit(...)`
