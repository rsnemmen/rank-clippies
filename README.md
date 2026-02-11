# LLM Benchmark Aggregator

A command-line tool that computes a unified model ranking from multiple benchmark leaderboards using **percentile-normalized** scores. It ingests a simple data file of per-benchmark ranks, normalizes them to a common 0–1 scale, and outputs a single sorted table so you can compare models at a glance.

**New:** Visualize model performance with statistical tiering! Generate scatter plots showing performance vs. cost, with models color-coded by statistical performance tiers based on the "Indistinguishable from Best" method.

---

## Why percentile normalization?

Raw ranks from different benchmarks are not directly comparable. A rank of 10 on a 300-model leaderboard is far more impressive than a rank of 10 on a 30-model leaderboard. Dividing each rank by the total number of evaluated models (`known_totals`) maps every score onto a **0–1 fractional percentile** (0 = best, 1 = worst), making cross-benchmark comparison meaningful.

---

## Quick start

```bash
python rank_models.py                     # reads ./ranks_general.txt by default
python rank_models.py my_data.txt         # or pass a custom file
python rank_models.py ranking_sample.txt  # run with example data
python rank_models.py ranking_sample.txt --plot  # generate performance plot
```

If the specified file does not exist, the script will print a helpful error message and exit.

### Requirements

**Core functionality:** Python 3.10+ (standard library only)

**Plotting feature (optional):** pandas, matplotlib, numpy
```bash
pip install pandas matplotlib numpy  # only needed for --plot flag
```

### Command-line Arguments

```
python rank_models.py [filename] [-p|--plot]
```

| Argument | Description |
|----------|-------------|
| `filename` | Path to the input file containing benchmark data. Defaults to `ranks_general.txt` if not provided. |
| `-p`, `--plot` | Generate a PNG scatter plot of model performance vs. cost with statistical tiering visualization. |

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

### ASCII Table

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

### Visualization Plot (`--plot`)

When using the `--plot` flag, the tool generates a PNG image with the same basename as your input file (e.g., `ranks_general.txt` → `ranks_general.png`).

#### Plot Features

- **Scatter plot** of model performance (Y-axis: Average Score) vs. cost (X-axis: Credit Cost per 1k tokens)
- **Log scale** on X-axis for better visualization of cost differences
- **Inverted Y-axis** so the best performers (lowest scores) appear at the top
- **Error bars** showing ±1 standard deviation for each model
- **Statistical tiering** using the "Indistinguishable from Best" method

#### Statistical Tiering Methodology

Models are grouped into performance tiers using a statistically rigorous approach:

1. **Tier 1**: Contains the best-performing model (leader) plus any models whose performance is statistically indistinguishable from the leader (68% confidence interval overlap)
2. **Tier 2**: After removing Tier 1 models, the next-best performer becomes the new leader; models overlapping with this leader form Tier 2
3. **Repeat**: Continue until all models are categorized

**Mathematical criterion**: A model belongs to the current tier if:
```
(model_score - 1σ) ≤ (leader_score + 1σ)
```

This means if a model's best-case performance (score minus one standard deviation) could reach the leader's worst-case performance (score plus one standard deviation), they are statistically tied.

#### Plot Interpretation

- **Colors**: Each tier has a distinct color from the tab10 colormap
- **Legend**: Shows which color corresponds to Tier 1, Tier 2, etc.
- **Tier 1** (top of plot): Best performers - models statistically tied for top rank
- **Higher tier numbers**: Progressively worse performance groups
- **Error bars**: Longer bars indicate more variable performance across benchmarks

Models evaluated on only 1 benchmark (no calculated SD) are assigned the average SD from all other models, allowing them to participate in tiering.

---

## Project structure

```
.
├── rank_models.py       # Main script
├── ranks_general.txt    # Default data file (input)
├── ranking_sample.txt   # Example data file
├── plotting.md          # Jupyter notebook reference for plotting
├── indistinguishable.md # Reference for tiering methodology
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

## Scientific Background

### Why "Indistinguishable from Best" Tiering?

Traditional rankings treat every rank position as meaningfully different. However, in statistical analysis, two models with overlapping confidence intervals cannot be confidently ranked against each other. The "Indistinguishable from Best" method, common in scientific literature, acknowledges this uncertainty by grouping statistically tied performers.

This approach answers the practical question: "Which models can I confidently say are among the best?" rather than forcing artificial distinctions where uncertainty exists.

### Confidence Interval Choice

We use z=1.0 (68% confidence interval, approximately ±1 standard deviation) as a balance between:
- **Strictness** (z=1.96, 95% CI): Creates many small tiers, potentially over-splitting
- **Leniency** (z=0.5): Creates few large tiers, potentially obscuring real differences

---

## License

This project is provided as-is for personal and research use.
