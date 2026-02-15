# Agent Guidelines for LLM_rank

## Project Overview
Python tool for computing unified model rankings from benchmark leaderboards using percentile-normalized scores. Features ASCII table output and optional visualization with statistical tiering based on the "Indistinguishable from Best" method.

**Core:** Single-file script with stdlib-only dependencies  
**Optional:** Plotting features require pandas, matplotlib, numpy

## Build/Lint/Test Commands

```bash
# Run the script
python rank_models.py                              # default: ranks_general.txt
python rank_models.py my_data.txt                  # custom input file
python rank_models.py my_data.txt --plot           # generate PNG visualization
python rank_models.py my_data.txt -d               # show tiering debug output
python rank_models.py my_data.txt -d -p            # debug + plot

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
python rank_models.py [filename] [-p|--plot] [-d|--debug]
```

- `filename` - Input file (default: `ranks_general.txt`)
- `--plot`, `-p` - Generate PNG visualization with tiering
- `--debug`, `-d` - Show detailed tiering diagnostics

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
10. Add debug output for complex algorithms when appropriate

## Output Standards

- **ASCII Table:** Fixed-width columns with `str.ljust()`/`rjust()`
- **Percentages:** 3 decimal places (0.XXX format)
- **Plots:** 150 DPI PNG, log scale X-axis, inverted Y-axis
- **Legend:** Show tiers when plotting enabled
- **Debug Output:** Use emojis and clear separators for readability

## Debug Mode Guidelines

When implementing debug functionality:
- Use conditional printing based on a `debug: bool` parameter
- Structure output with clear headers and separators
- Include all intermediate calculation steps
- Show before/after states for transformations
- Use visual indicators (✓, ✗, 👑, 📊) for quick scanning
- Print to stdout (not stderr) for debug output
- Keep debug logic separate from core algorithm when possible
