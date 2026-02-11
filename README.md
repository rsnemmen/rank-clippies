# LLM Benchmark Aggregator

A command-line tool that computes a unified model ranking from multiple benchmark leaderboards using **percentile-normalized** scores. It ingests a simple data file of per-benchmark ranks, normalizes them to a common 0–1 scale, and outputs a single sorted table so you can compare models at a glance.

---

## Why percentile normalization?

Raw ranks from different benchmarks are not directly comparable. A rank of 10 on a 300-model leaderboard is far more impressive than a rank of 10 on a 30-model leaderboard. Dividing each rank by the total number of evaluated models (`known_totals`) maps every score onto a **0–1 fractional percentile** (0 = best, 1 = worst), making cross-benchmark comparison meaningful.

---

## Quick start

```bash
python rank_models.py                     # reads ./ranks_general.txt by default
python rank_models.py my_data.txt         # or pass a custom file
python rank_models.py ranking_sample.txt  # run with example data
```

If the specified file does not exist, the script will print a helpful error message and exit.

### Requirements

Python 3.10+ (standard library only — no third-party packages).

### Command-line Arguments

```
python rank_models.py [filename]
```

- **`filename`** (optional): Path to the input file containing benchmark data. Defaults to `ranks_general.txt` if not provided.

### Error Handling

If the specified file cannot be found, the script outputs:

```
Error: File 'filename.txt' not found.

Usage: python rank_models.py [filename]
  filename  Path to input file (default: ranks_general.txt)

Run with 'ranking_sample.txt' for an example:
  python rank_models.py ranking_sample.txt
```

---

## Input format — `ranks_general.txt`

The file contains one or more **benchmark dictionaries** followed by exactly one **cost dictionary** as the last entry. Each dictionary is a standard Python `dict` literal and may span multiple lines.

### Benchmark dictionaries

Each benchmark is written as `name={...}` (the name is purely cosmetic and not used in output). Every dictionary must include a `known_totals` key indicating how many models were evaluated on that leaderboard. All other keys are model identifiers mapped to either an integer rank or `None` (not evaluated).

```python
LiveBench={"sonnet":12, "opus":1, "haiku":41,
"gpt":3, "gemini":6,
"known_totals":52}
```

### Cost dictionary

The **last** dictionary in the file holds credit costs per 1 000 tokens. It has no `known_totals` key and needs no variable-name prefix. Models missing from this dictionary will show `N/A` in the output.

```python
# Credit cost per 1k tokens
{"sonnet":500, "opus":850, "haiku":170, "gpt":470, "gemini":370}
```

### Comments

Lines whose first non-whitespace character is `#` are ignored.

---

## Methodology

### 1. Percentile normalization

For every model with a non-`None` rank in a given benchmark:

```
percentile = rank / known_totals
```

### 2. Averaging

A model's base score is the arithmetic mean of its percentile values across all benchmarks it appears in. Every benchmark carries equal weight.

### 3. Sparse-data penalty

Models evaluated on very few benchmarks get a penalty added to their average to avoid over-rewarding thin evidence:

| Benchmarks appeared in | Penalty |
|------------------------|---------|
| 1                      | +0.25   |
| 2                      | +0.10   |
| 3 or more              | +0.00   |

### 4. Standard deviation

Population standard deviation is computed over the pre-penalty percentile scores. Models with fewer than 2 data points report `N/A`.

### 5. Missing data

`None` values and absent keys are treated as "not evaluated" — the benchmark is simply excluded from that model's calculations.

### 6. Name matching

Model keys are compared as exact strings. No fuzzy matching or alias merging is performed (e.g. `qwen3-235` and `qwen3-80` remain separate models).

---

## Output

A sorted ASCII table (best model first):

```
+-----------------------------------------------------------------------------+
| Rank  | Model                  | Avg Pctl | Std Dev  | # Benchmarks | Cost/1k |
+-----------------------------------------------------------------------------+
| 1     | opus                   | 0.019    | 0.009    | 4            | 850     |
| 2     | gemini                 | 0.048    | 0.046    | 4            | 370     |
| ...   | ...                    | ...      | ...      | ...          | ...     |
+-----------------------------------------------------------------------------+
```

| Column         | Description                                              |
|----------------|----------------------------------------------------------|
| **Rank**       | Position in the final aggregated ranking (1 = best).     |
| **Model**      | Model identifier string (exactly as written in the file).|
| **Avg Pctl**   | Mean percentile after sparse-data penalty, 0–1 scale.    |
| **Std Dev**    | Population std dev of percentile scores, or `N/A`.       |
| **# Benchmarks** | Number of benchmarks the model was evaluated on.      |
| **Cost/1k**    | Credit cost per 1 000 tokens, or `N/A` if unavailable.   |

---

## Project structure

```
.
├── rank_models.py       # Main script
├── ranks_general.txt    # Default data file (input)
├── ranking_sample.txt   # Example data file
└── README.md            # Documentation
```

---

## Extending the data

To add a new benchmark, append a new dictionary **above** the cost dictionary in your data file:

```python
new_bench={"opus":2, "sonnet":5, "gpt":1, "known_totals":40}
```

To add a new model, insert its key and rank into each relevant benchmark dictionary and, optionally, add its cost to the cost dictionary.

---

## License

This project is provided as-is for personal and research use.