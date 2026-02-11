# Agent Guidelines for LLM_rank

## Project Overview
Simple Python tool for computing unified model rankings from benchmark leaderboards using percentile-normalized scores. Single-file script with no external dependencies.

## Build/Lint/Test Commands

Since this is a standard library-only Python script:

```bash
# Run the script
python rank_models.py                    # uses ranking.txt
python rank_models.py my_data.txt        # custom input file

# Type checking (if mypy available)
mypy rank_models.py --strict

# Linting (if ruff/flake8 available)
ruff check rank_models.py

# Formatting (if ruff/black available)
ruff format rank_models.py

# Running tests (when added)
python -m pytest tests/ -v              # all tests
python -m pytest tests/test_file.py -v  # single test file
python -m pytest tests/test_file.py::test_function -v  # single test
```

**Note**: Currently no tests exist. When adding tests, use pytest and place in `tests/` directory.

## Code Style Guidelines

### Python Version
- Python 3.10+ (uses `dict[str, ...]` syntax, not `Dict[str, ...]`)

### Imports
- Group: stdlib imports only (no third-party dependencies)
- Format: `import module` before `from module import name`
- One blank line between import groups
- Example:
  ```python
  import ast
  import math
  import sys
  ```

### Formatting
- Indent: 4 spaces (no tabs)
- Line length: ~100 characters max
- Quotes: Double quotes for strings, single quotes for dict keys when needed
- Trailing commas in multi-line collections
- One blank line between function definitions
- Two blank lines before top-level functions/classes

### Type Hints
- Use inline variable annotations: `model_scores: dict[str, list[float]]`
- Use `|` for unions (Python 3.10+): `str | None`
- Annotate function parameters and return types

### Naming Conventions
- Functions/variables: `snake_case`
- Constants: `UPPER_CASE` (e.g., `KNOWN_TOTALS`)
- Modules: `lowercase` with underscores if needed
- Type variables: descriptive, not single letters

### Error Handling
- Use specific exception types: `except (ValueError, SyntaxError) as exc:`
- Print warnings to stderr: `print(..., file=sys.stderr)`
- Exit on fatal errors: `sys.exit("Error: ...")`
- Handle `None` explicitly with checks, not try/except

### Documentation
- Module docstring at top with usage example
- Function docstrings in imperative mood: `"Parse file and return..."`
- Inline comments for non-obvious logic
- Comments should explain "why", not "what"

### Code Patterns
- Prefer list/dict comprehensions for simple transforms
- Use `dict.get()` with defaults for optional keys
- Sort using `list.sort(key=lambda x: x[0])` for clarity
- Use f-strings for formatting: `f"{value:.3f}"`

### File I/O
- Use context managers: `with open(...) as f:`
- Handle encoding explicitly when non-ASCII expected
- Validate file existence before parsing

### Output
- Table formatting: Use fixed-width columns with `str.ljust()`/`rjust()`
- Progress/info to stdout, errors to stderr
- Decimal precision: 3 places for percentages (0.XXX format)

## Project Structure
```
.
├── rank_models.py      # Main script (executable)
├── ranking.txt         # Data file (input)
├── ranking_sample.txt  # Example data
├── README.md           # Documentation
└── tests/              # Tests (to be added)
```

## Adding Features
1. Maintain zero third-party dependencies
2. Keep backward compatibility with existing data format
3. Update README.md when changing CLI interface
4. Add type hints to all new code
5. Follow existing ASCII table formatting style for output
