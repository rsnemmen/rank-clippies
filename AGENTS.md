# Agent Guidelines for LLM_rank

## Project Overview
Python tool for computing unified model rankings from benchmark leaderboards using percentile-normalized scores. Features ASCII table output and optional visualization with statistical tiering based on the "Indistinguishable from Best" method.

**Core:** Single-file script with stdlib-only dependencies  
**Optional:** Plotting features require pandas, matplotlib, numpy

## Build/Lint/Test Commands

```bash
# Run the script
python rank_models.py                    # default: ranks_general.txt
python rank_models.py my_data.txt        # custom input file
python rank_models.py my_data.txt --plot # generate PNG visualization

# Type checking
mypy rank_models.py --strict

# Linting
ruff check rank_models.py

# Formatting
ruff format rank_models.py

# Testing (when added)
python -m pytest tests/ -v                           # all tests
python -m pytest tests/test_file.py -v               # single test file
python -m pytest tests/test_file.py::test_function -v # single test
```

**Note:** No tests exist yet. When adding, use pytest in `tests/` directory.

## CLI Usage

```bash
python rank_models.py [filename] [-p|--plot]
```

- `filename` - Input file (default: `ranks_general.txt`)
- `--plot`, `-p` - Generate PNG visualization with tiering

## Code Style Guidelines

### Python Version
- Python 3.10+ (uses `dict[str, ...]` and `|` union syntax)

### Imports
- **Core:** stdlib imports only (argparse, ast, math, os, sys)
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

### Naming Conventions
- Functions/variables: `snake_case`
- Constants: `UPPER_CASE`
- Modules: `lowercase`
- Type variables: descriptive names

### Documentation
- Module docstring with usage examples
- Function docstrings in imperative mood
- Inline comments explain "why", not "what"

### Code Patterns
- Prefer comprehensions for simple transforms
- Use `dict.get()` with defaults
- Use f-strings: `f"{value:.3f}"`
- Use `sys.exit()` for fatal errors
- Handle `None` explicitly with checks, not try/except

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

## Project Structure

```
.
├── rank_models.py          # Main script (executable)
├── ranks_general.txt       # Default data file
├── ranking_sample.txt      # Example data file
├── ranks_coding.txt        # Coding-specific benchmarks
├── plotting.md             # Plotting reference (Jupyter notebook)
├── indistinguishable.md    # Tiering methodology reference
├── README.md               # User documentation
├── AGENTS.md               # This file
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

## Output Standards

- **ASCII Table:** Fixed-width columns with `str.ljust()`/`rjust()`
- **Percentages:** 3 decimal places (0.XXX format)
- **Plots:** 150 DPI PNG, log scale X-axis, inverted Y-axis
- **Legend:** Show tiers when plotting enabled
