# Agent Guidelines for LLM_rank

## Project Overview
Python tool for computing unified model rankings from benchmark leaderboards using percentile-normalized scores. Features ASCII table output and optional visualization with statistical tiering based on the "Indistinguishable from Best" method.

**Repository:** https://github.com/rsnemmen/rank-clippies

**Core:** Single-file script with stdlib-only dependencies  
**Optional:** Plotting features require pandas, matplotlib, numpy

## Methodology

### Percentile Normalization
Different leaderboards evaluate different numbers of models. Being ranked 5th out of 600 models is more impressive than 5th out of 30. To make cross-benchmark comparisons fair, normalize each rank to a fractional percentile:

```
percentile = rank / total_models_evaluated
```

This puts every score on a 0-1 scale (0 = best, 1 = worst).

### Aggregation and Penalties
A model's composite score is the unweighted median of its percentiles across all benchmarks it appears on. The median is preferred over the mean for robustness against outlier benchmarks. To avoid boosting models that score well on only a single benchmark, a sparse-data penalty is added:
- +0.25 for one benchmark
- +0.10 for two benchmarks
- 0 for three or more

### Statistical Tiering
Models are grouped into tiers using the "indistinguishable from best" method:
1. The best model becomes the tier leader
2. Every remaining model whose ±semi-IQR interval overlaps with the leader's interval joins the same tier
3. Remove the tier, promote the next-best model to leader, and repeat

Model i is grouped with leader j if:
```
score_i + IQR_half_i >= score_j - IQR_half_j
```

Semi-IQR (half the interquartile range) is the dispersion measure for error bars and tiering — the natural robust companion to the median aggregate. Models evaluated on fewer than three benchmarks have no semi-IQR (IQR is degenerate for n<3), so the average semi-IQR across all other models is used as a stand-in.

### Cost Metric
Use credit cost per 1,000 tokens from a single API provider (e.g., Poe) as a uniform pricing reference for consistent relative comparisons. In both the ASCII table and the scatter plot, costs are normalized so the best-ranked model = 1.000; all other values are multiples of that baseline.

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

# Testing (when added)
python -m pytest tests/ -v                         # all tests
python -m pytest tests/test_file.py -v             # single test file
python -m pytest tests/test_file.py::test_function -v  # single test
```

**Note:** No tests exist yet. When adding, use pytest in `tests/` directory.

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
- **Core:** stdlib imports only (argparse, ast, math, os, sys, typing)
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
├── data/
│   ├── benchmarks.txt      # All benchmarks with category tags (single source of truth)
│   ├── models.txt          # All model metadata: cost and open-weight flag
│   └── rank_convert.nb     # Wolfram Mathematica notebook for data conversion
├── figures/                # Generated PNG plots
├── README.md               # User documentation
├── AGENTS.md               # This file
├── LICENSE                 # MIT license
└── tests/                  # Tests (to be added)
```

## Adding Features

1. Keep stdlib-only for core functionality
2. Optional features can use pandas/matplotlib/numpy (import inside functions)
3. Maintain backward compatibility with existing data format
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
- **Plots:** 150 DPI PNG, log scale X-axis (cost normalized to best model = 1), inverted Y-axis
- **Legend:** Show tiers when plotting enabled
- **Quadrants (`-q`):** Divides the scatter plot into four regions using the geometric mean of cost (X) and median score (Y) as midpoints; regions are shaded and labelled "Best value" (low cost, high perf), "Premium" (high cost, high perf), "Budget" (low cost, low perf), "Avoid" (high cost, low perf)
- **Debug Output:** Use emojis and clear separators for readability

## Debug Mode Guidelines

When implementing debug functionality:
- Use conditional printing based on a `debug: bool` parameter
- The `z_score` parameter controls confidence level (1.0 = 68% CI, 1.96 = 95% CI)
- Structure output with clear headers and separators
- Include all intermediate calculation steps
- Show before/after states for transformations
- Use visual indicators (✓, ✗, 👑, 📊) for quick scanning
- Print to stdout (not stderr) for debug output
- Keep debug logic separate from core algorithm when possible
