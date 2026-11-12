# Agent Guidelines for LLM_rank

## Project Overview
Python tool for computing unified model rankings from benchmark leaderboards using gap-to-best normalized scores. Features ASCII table output and optional visualization with statistical tiering based on the "Indistinguishable from Best" method.

**Repository:** https://github.com/rsnemmen/rank-clippies

**Core:** Single-file script with stdlib-only dependencies, reading TOML data via `tomllib`  
**Optional:** Plotting features require pandas, matplotlib, numpy

## Methodology

### Normalization
Every current benchmark is score-based. Each score becomes its gap to the best tracked model as a share of the benchmark's range:

```
gap = (max_score - score) / (max_score - min_score)
```

0 = best tracked model, 1 = the floor. This is not a percentile, and a new top model shifts everyone's gap. `min_score` should be the lowest score on the full source leaderboard; `load_data` exits if any score is below it. Rank-based benchmarks (`known_totals`) use `rank / known_totals`, a true percentile; ranks must lie in `1..known_totals`.

### Aggregation and Coverage Shrinkage
A model's raw score is the unweighted median of its gaps across all benchmarks it appears on. The median is preferred over the mean for robustness against outlier benchmarks. To avoid over-rewarding thin evidence, the median is blended with a below-average phantom result worth `SHRINK_K = 1` benchmark:

```
score = (n × median + SHRINK_K × prior) / (n + SHRINK_K)
```

`prior` (`shrinkage_prior`) is the `SHRINK_QUANTILE = 0.75` quantile of all models' medians in the same benchmark set, so it is recomputed in every bootstrap resample. Own-data weight is n/(n+1): 50% at n=1, 75% at n=3, 91% at n=10. k is deliberately fixed: the data-estimated σ²/τ² ratio swings from 0.3 to 13 across history snapshots.

Asymmetric IQR bounds (`median − Q1`, `Q3 − median`), scaled by the same n/(n+k) weight, are shown as error bars. They measure spread across benchmarks, not uncertainty in the median. Models with n<3 have none.

### Uncertainty and Tiering (benchmark bootstrap)
Implemented in `ranking_math.py`. `bootstrap_scores` resamples the category's benchmarks with replacement (`N_BOOT = 2000`, fixed seed) and reruns `calculate_rankings` on each draw; all models share each draw, so comparisons are paired.
- `rank_intervals`: the middle 68% of each model's bootstrap ranks, widened if needed to include its listed rank.
- `assign_tiers`: the best remaining model leads a tier; walking down the ranking, each model joins while `P(score_i ≤ score_leader) ≥ TIER_ALPHA` (0.16) over resamples where both have data; the first that fails leads the next tier. Tiers are therefore contiguous in rank order.

### Cost Metric
Use USD per 1M tokens (input + output combined) as the cost metric for consistent relative comparisons across providers. In both the ASCII table and the scatter plot, costs are normalized so the best-ranked model = 1.000; all other values are multiples of that baseline.

### Benchmark Sources
**General reasoning:** LiveBench, Arena, Artificial Analysis Intelligence Index, Scale's Humanity's Last Exam, SimpleBench, MMMLU, GPQA Diamond, CharXiv, NYT Connections

**Coding and agentic coding:** LiveBench (coding), Arena (coding), Artificial Analysis, SWE-bench Verified, SWE-bench Pro Public, SWEatlas Codebase Q&A, Terminal-Bench 2.0

**Agentic and computer use:** OSWorld, Terminal-Bench 2.0, Artificial Analysis Agentic Index, SWE-bench Verified, SWE-bench Pro Public

**Math and STEM expert reasoning:** GPQA Diamond, Humanity's Last Exam, USA Math Olympiad (USAMO)

## Build/Lint/Test Commands

```bash
# Run the script  (valid categories: general, coding, agentic, stem)
python rank_models.py                              # default: coding
python rank_models.py general                      # general intelligence ranking
python rank_models.py coding --plot               # generate PNG visualization
python rank_models.py agentic -d                  # show tiering debug output
python rank_models.py stem -d -p                  # debug + plot
python rank_models.py general -p -q               # plot with quadrant overlays

# Type checking
mypy rank_models.py --strict

# Linting
ruff check rank_models.py                          # check for issues
ruff check rank_models.py --fix                    # auto-fix issues

# Formatting
ruff format rank_models.py                         # format code

# Testing
python -m pytest tests/ -v                         # all tests
python -m pytest tests/test_file.py -v             # single test file
python -m pytest tests/test_file.py::test_function -v  # single test

# Full validation
make validate                                      # ruff + mypy --strict + pytest + JSON export smoke test
```

Tests live in `tests/` and cover ranking math, coverage shrinkage, bootstrap tiering, TOML loader errors, and website JSON export shape.

## CLI Usage

```bash
python rank_models.py [category] [-p|--plot] [-d|--debug] [-q|--quadrants]
python rank_models.py -h                          # Show help
```

- `category` - Ranking category: `general`, `coding`, `agentic`, `stem` (default: `coding`)
- `--plot`, `-p` - Generate PNG visualization with tiering
- `--debug`, `-d` - Show detailed tiering diagnostics
- `--quadrants`, `-q` - Overlay quadrant dividers and labels on the scatter plot (requires `--plot`)

## Code Style Guidelines

### Python Version
- Python 3.10+ (uses `dict[str, ...]` and `|` union syntax)

### Imports
- **Core:** stdlib imports only (argparse, datetime, json, math, os, subprocess, sys, tomllib, typing)
- **Optional plotting:** pandas, matplotlib, numpy imported inside functions
- Sort alphabetically within groups
- One blank line between groups

### Formatting
- Indent: 4 spaces (no tabs)
- Line length: ~100 characters
- Quotes: Double for strings, single for dict keys when needed
- Trailing commas in multi-line collections
- One blank line between functions
- Two blank lines before top-level functions

### Type Hints
- Use inline annotations: `model_scores: dict[str, list[float]]`
- Use `|` for unions: `str | None`
- Annotate all function parameters and return types
- Use descriptive parameter names in docstrings

### Naming Conventions
- Functions/variables: `snake_case`
- Constants: `UPPER_CASE`
- Modules: `lowercase`
- Type variables: descriptive names
- Boolean flags: use `is_` or `has_` prefix when appropriate

### Documentation
- Module docstring with usage examples
- Function docstrings in imperative mood ("Calculate", "Parse", not "Calculates")
- Inline comments explain "why", not "what"
- Keep docstrings concise but complete

### Code Patterns
- Prefer comprehensions for simple transforms
- Use `dict.get()` with defaults
- Use f-strings: `f"{value:.3f}"`
- Use `sys.exit()` for fatal errors
- Handle `None` explicitly with checks, not try/except
- Use early returns to reduce nesting

### File I/O
- Use context managers: `with open(...) as f:`
- Validate file existence before parsing
- Print errors to stderr, info to stdout

## Error Handling

- **Warnings:** `print(..., file=sys.stderr)`
- **Fatal errors:** `sys.exit("Error: ...")` or `sys.exit(1)`
- **Exceptions:** Catch specific types: `except (ValueError, SyntaxError) as exc:`

### File Validation Pattern

```python
if not os.path.exists(filename):
    print(f"Error: File '{filename}' not found.", file=sys.stderr)
    print("\nUsage: python rank_models.py [filename]", file=sys.stderr)
    sys.exit(1)
```

### Optional Dependencies Pattern

```python
try:
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError as exc:
    print(f"Error: {exc}", file=sys.stderr)
    print("Plotting requires pandas and matplotlib.", file=sys.stderr)
    sys.exit(1)
```

## Project Structure

```
.
├── rank_models.py          # Main script (executable)
├── ranking_math.py         # Shared normalization, median aggregation, bootstrap tiering
├── data/
│   ├── benchmarks.toml     # All benchmarks with category tags (single source of truth)
│   └── models.toml         # All model metadata: cost and open-weight flag
├── docs/
│   ├── app.js              # Static website logic
│   ├── data/               # Generated JSON consumed by the website
│   └── index.html
├── figures/                # Generated PNG plots (ignored)
├── README.md               # User documentation
├── AGENTS.md               # This file
├── LICENSE                 # MIT license
└── tests/
    └── test_rank_models.py # Ranking, loader, and JSON export tests
```

## Adding Features

1. Keep stdlib-only for core functionality
2. Optional features can use pandas/matplotlib/numpy (import inside functions)
3. Maintain backward compatibility with the existing TOML data format
4. Update README.md for CLI changes
5. Add type hints to all new code
6. Follow existing ASCII table formatting
7. Print errors to stderr, not stdout
8. Include file validation for new file operations
9. Graceful handling when optional dependencies missing
10. Add debug output for complex algorithms when appropriate

## Output Standards

- **ASCII Table:** Fixed-width columns with `str.ljust()`/`rjust()`
- **Percentages:** 3 decimal places (0.XXX format)
- **Plots:** 300 DPI PNG, log scale X-axis (cost normalized to best model = 1), inverted Y-axis
- **Legend:** Show tiers when plotting enabled
- **Quadrants (`-q`):** Divides the scatter plot into four regions using the geometric mean of cost (X) and median score (Y) as midpoints; regions are shaded and labelled "Best value" (low cost, high perf), "Premium" (high cost, high perf), "Budget" (low cost, low perf), "Avoid" (high cost, low perf)
- **Debug Output:** Use emojis and clear separators for readability

## Debug Mode Guidelines

When implementing debug functionality:
- Use conditional printing based on a `debug: bool` parameter
- The `alpha` parameter of `categorize_tiers` (default `TIER_ALPHA = 0.16`) is the minimum share of bootstrap resamples in which a candidate must match the tier leader
- Structure output with clear headers and separators
- Include all intermediate calculation steps
- Show before/after states for transformations
- Use visual indicators (✓, ✗, 👑, 📊) for quick scanning
- Print to stdout (not stderr) for debug output
- Keep debug logic separate from core algorithm when possible
