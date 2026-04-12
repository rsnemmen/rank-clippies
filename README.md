# LLM Benchmark Aggregator

A command-line tool that computes a unified model ranking from multiple benchmark leaderboards using **percentile-normalized** scores. It ingests a simple data file of per-benchmark ranks, normalizes them to a common 0–1 scale, and outputs a single sorted table so you can compare models at a glance.

Visualize model performance with statistical tiering: Generate scatter plots showing performance vs. cost, with models color-coded by statistical performance tiers based on the "Indistinguishable from Best" method.

## Quick start

```shell
# evaluate across general reasoning leaderboards 
python rank_models.py data/general.txt  
```

The above command will produce the following output:

```text
+-------------------------------------------------------------------------------------+
| Rank  | Model                   | Avg Pctl  | IQR/2     | # Benchmarks  | Rel. Cost |
+-------------------------------------------------------------------------------------+
| 1     | opus                    | 0.030     | 0.018     | 7             | 1.000     |
| 2     | gemini                  | 0.076     | 0.044     | 7             | 0.435     |
| 3     | gpt                     | 0.077     | 0.041     | 7             | 0.553     |
| 4     | gpt-codex               | 0.095     | 0.063     | 6             | 0.553     |
| 5     | gpt-pro                 | 0.103     | 0.091     | 3             | 6.588     |
| 6     | sonnet                  | 0.170     | 0.121     | 7             | 0.588     |
| 7     | kimi                    | 0.205     | 0.115     | 7             | 0.141     |
| 8     | glm                     | 0.226     | 0.198     | 7             | 0.111     |
| 9     | gemini-flash            | 0.237     | 0.209     | 7             | 0.118     |
| 10    | haiku                   | 0.241     | 0.183     | 5             | 0.200     |
...
```

For the plots:

```shell
# generate performance plot for general reasoning
python rank_models.py data/general.txt --plot

# generate scatter plot with quadrant overlays
python rank_models.py data/general.txt --plot --quadrants
```

This will output two plots. The first is the average ranking as a function of API cost:
![](figures/general.png)  
**Figure 1: General intelligence vs model cost.** Y-axis indicates the median percentile rank on a scale from 1 (best) to 100 (worst). X-axis is the cost relative to the best-ranked model (log scale; best model = 1). Colors indicate the model tier. Error bars (±semi-IQR) indicate the variation of a model ranking across different benchmarks.

The second plot is a different visualization of the tiers: 
![](figures/general_ranking.png)  
**Figure 2: Model ranking (general intelligence).** Bigger circles indicate more expensive models. Small semi-transparent dots show the individual per-benchmark percentile values, revealing the spread, skewness, and outliers behind each aggregate score.

The same plots but evaluating coding and agentic coding performance are available in the `data` folder (files `coding.png` and `coding_ranking.png`).

For more detailed information about the ranking procedure and for debugging:

```
# show detailed tiering diagnostics
python rank_models.py data/general.txt --debug

# combine debug and plot
python rank_models.py data/general.txt -d -p
```


### Requirements

**Core functionality:** Python 3.10+ (standard library only)

**Plotting feature (optional):** pandas, matplotlib, numpy
```bash
pip install pandas matplotlib numpy  # only needed for --plot flag
```

### Command-line Arguments

```
python rank_models.py [filename] [-p|--plot] [-d|--debug] [-q|--quadrants]
```

| Argument | Description |
|----------|-------------|
| `filename` | Path to the input file containing benchmark data. Defaults to `ranks_general.txt` if not provided. |
| `-p`, `--plot` | Generate a PNG scatter plot of model performance vs. cost with statistical tiering visualization. |
| `-d`, `--debug` | Show detailed tiering diagnostics including leader selection, overlap checks, and tier assignments. |
| `-q`, `--quadrants` | Overlay quadrant dividers and labels on the scatter plot. Divides the chart into four regions — **Best value** (low cost, high perf), **Premium** (high cost, high perf), **Budget** (low cost, low perf), **Avoid** (high cost, low perf) — using the geometric mean of cost and median score as midpoints. Requires `--plot`. |

## Input format for datafile

The file contains one or more **benchmark dictionaries** followed by exactly one **cost dictionary** as the last entry. Each dictionary is a standard Python `dict` literal and may span multiple lines.

### Benchmark dictionaries

Each benchmark is written as `name={...}` (the name is purely cosmetic and not used in output). There are two forms:

**Rank-based** — models are mapped to integer ranks (lower = better). `known_totals` is required and indicates how many models were evaluated on that leaderboard. Non-evaluated models are `None`.

```python
LiveBench={"sonnet":12, "opus":1, "haiku":41,
"gpt":3, "gemini":6,
"known_totals":52}
```

**Score-based** — models are mapped to numeric scores (higher = better). `min_score` is required and sets the floor for percentile normalization. `known_totals` is optional and ignored.

```python
HLE={"sonnet":50.1, "opus":72.5, "haiku":18.3,
"gpt":61.2, "gemini":55.8,
"min_score":2.72}
```

The included files contain leaderboards which are current as of Feb. 11 2026 manually pulled from the following websites:

https://livebench.ai/#/  
https://scale.com/leaderboard/  
https://arena.ai/leaderboard/  
https://www.tbench.ai/leaderboard/terminal-bench/2.0  
https://artificialanalysis.ai  
https://gorilla.cs.berkeley.edu/leaderboard.html#leaderboard  
https://www.swebench.com

### Cost dictionary

The last dictionary in the file holds credit costs per 1 000 tokens. It has no `known_totals` key and needs no variable-name prefix. Models missing from this dictionary will show `N/A` in the output. The value here is arbitrary. In the sample data included, it corresponds to Poe API credits.

```python
# Credit cost per 1k tokens
{"sonnet":500, "opus":850, "haiku":170, "gpt":470, "gemini":370}
```


## Methodology

### 1. Percentile normalization

For every model with a non-`None` rank in a given benchmark:

```
percentile = rank / known_totals
```

**Why percentile normalization?**  
Raw ranks from different benchmarks are not directly comparable. A rank of 10 on a 300-model leaderboard is far more impressive than a rank of 10 on a 30-model leaderboard. Dividing each rank by the total number of evaluated models (`known_totals`) maps every score onto a 0–1 fractional percentile (0 = best, 1 = worst), making cross-benchmark comparison meaningful.

### 2. Median aggregation

A model's base score is the median of its percentile values across all benchmarks it appears in. Every benchmark carries equal weight. The median is used instead of the mean for robustness against outlier benchmarks where a model performs unusually well or poorly.

### 3. Sparse-data penalty

Models evaluated on very few benchmarks get a penalty added to their average to avoid over-rewarding thin evidence:

| Benchmarks appeared in | Penalty |
|------------------------|---------|
| 1                      | +0.25   |
| 2                      | +0.10   |
| 3 or more              | +0.00   |

**Note:** Penalized scores are capped at 1.0 (100th percentile) to prevent exceeding the theoretical maximum.

### 4. Semi-IQR (dispersion)

The semi-IQR (half the interquartile range) is computed over the pre-penalty percentile scores as the robust dispersion measure for error bars and statistical tiering — the natural companion to the median aggregate. Models with fewer than 3 data points report `N/A` (IQR is degenerate for n < 3); these use the average semi-IQR across all other models as a stand-in for tiering purposes.

### 5. Missing data

`None` values and absent keys are treated as "not evaluated" — the benchmark is simply excluded from that model's calculations.

### 6. Name matching

Model keys are compared as exact strings. No fuzzy matching or alias merging is performed (e.g. `qwen3-235` and `qwen3-80` remain separate models).


## Output

### ASCII Table

A sorted ASCII table (best model first):

```
+------------------------------------------------------------------------------+
| Rank  | Model                  | Avg Pctl | IQR/2    | # Benchmarks | Rel. Cost |
+------------------------------------------------------------------------------+
| 1     | opus                   | 0.019    | 0.008    | 4            | 1.000     |
| 2     | gemini                 | 0.048    | 0.041    | 4            | 0.435     |
| ...   | ...                    | ...      | ...      | ...          | ...       |
+------------------------------------------------------------------------------+
```

| Column           | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| **Rank**         | Position in the final aggregated ranking (1 = best).                        |
| **Model**        | Model identifier string (exactly as written in the file).                   |
| **Avg Pctl**     | Median percentile after sparse-data penalty, 0–1 scale.                     |
| **IQR/2**        | Semi-IQR (half the interquartile range) of percentile scores, or `N/A`.     |
| **# Benchmarks** | Number of benchmarks the model was evaluated on.                            |
| **Rel. Cost**    | Cost relative to the best-ranked model (best = 1.000), or `N/A` if unavailable. |

### Visualization Plot (`--plot`)

When using the `--plot` flag, the tool generates a PNG image with the same basename as your input file (e.g., `general.txt` → `general.png`).

**Quadrant overlay (`--quadrants`, `-q`):** Pass this flag together with `--plot` to divide the scatter plot into four labelled, shaded regions. The vertical divider is placed at the geometric mean of all plotted model costs (log-scale midpoint); the horizontal divider sits at the median aggregate score. This makes it easy to spot which models offer the best performance per dollar.

#### Statistical Tiering Methodology

Models are grouped into performance tiers using a statistically rigorous approach:

1. **Tier 1**: Contains the best-performing model (leader) plus any models whose ±semi-IQR interval overlaps with the leader's interval
2. **Tier 2**: After removing Tier 1 models, the next-best performer becomes the new leader; models overlapping with this leader form Tier 2
3. **Repeat**: Continue until all models are categorized

**Mathematical criterion**: A model belongs to the current tier if:
```
(model_score - semi-IQR) ≤ (leader_score + semi-IQR)
```

This means if a model's best-case performance (score minus semi-IQR) could reach the leader's worst-case performance (score plus semi-IQR), they are considered statistically tied. Semi-IQR is used as the dispersion measure because it is the natural robust companion to the median aggregate and makes no distributional assumptions.

## Extending the data

To add a new benchmark, append a new dictionary above the cost dictionary in your data file:

```python
new_bench={"opus":2, "sonnet":5, "gpt":1, "known_totals":40}
```

To add a new model, insert its key and rank into each relevant benchmark dictionary and, optionally, add its cost to the cost dictionary.


## Why "Indistinguishable from Best" Tiering?

Traditional rankings treat every rank position as meaningfully different. However, in statistical analysis, two models with overlapping confidence intervals cannot be confidently ranked against each other. The "Indistinguishable from Best" method, common in scientific literature, acknowledges this uncertainty by grouping statistically tied performers.

This approach answers the practical question: "Which models can I confidently say are among the best?" rather than forcing artificial distinctions where uncertainty exists.
