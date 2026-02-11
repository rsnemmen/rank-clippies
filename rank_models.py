#!/usr/bin/env python
"""
Compute a unified model ranking from multiple benchmark leaderboards
using percentile-normalized scores.

Usage:  python rank_models.py [ranking.txt]
        python rank_models.py [ranking.txt] --plot
"""

import argparse
import ast
import math
import os
import sys


def create_plot(results: list[tuple], output_filename: str) -> None:
    """Create a scatter plot of model performance vs. cost and save to PNG.
    
    Args:
        results: List of tuples (model, avg, sd, n, cost) from main computation
        output_filename: Path to save the PNG output
    """
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print("Plotting requires pandas and matplotlib. Install with:", file=sys.stderr)
        print("  pip install pandas matplotlib", file=sys.stderr)
        sys.exit(1)
    
    # Convert results to DataFrame
    df_data = []
    for model, avg, sd, n, cost in results:
        df_data.append({
            'Model Name': model,
            'Average Score': avg * 100,  # Convert from percentile to match table display
            'Std Dev': sd * 100 if sd is not None else 0.0,  # Convert sd to same scale
            'Credit Cost (per 1k)': cost
        })
    
    df = pd.DataFrame(df_data)
    
    # Convert numeric columns (handling None in Cost)
    df['Credit Cost (per 1k)'] = pd.to_numeric(df['Credit Cost (per 1k)'], errors='coerce')
    
    # Filter out rows with NaN costs for the plot
    plot_df = df.dropna(subset=['Credit Cost (per 1k)'])
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Create Scatter Plot with error bars
    plt.errorbar(plot_df['Credit Cost (per 1k)'], 
                 plot_df['Average Score'], 
                 yerr=plot_df['Std Dev'],
                 fmt='o',  # circle markers
                 color='royalblue', 
                 ecolor='royalblue',  # subtle gray error bars
                 alpha=0.7, 
                 markersize=10,
                 capsize=0,  # no caps on error bars
                 markeredgecolor='black')
    
    # Annotate points with Model Names
    for _, row in plot_df.iterrows():
        model_name: str = str(row['Model Name'])
        cost_val: float = float(row['Credit Cost (per 1k)'])
        score_val: float = float(row['Average Score'])
        plt.annotate(model_name, 
                     (cost_val, score_val), 
                     xytext=(5, 5), textcoords='offset points', fontsize=13)
    
    # Formatting
    plt.title('LLM Model Performance vs. Cost', fontsize=20)
    plt.ylabel('Average Score (Lower is Better)', fontsize=19)
    plt.xlabel('Credit Cost (per 1k tokens) - Log Scale', fontsize=19)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    
    # Use a log scale for X axis (Cost)
    plt.xscale('log')
    
    # Invert Y axis so the "Best" (Rank 1/Low Score) is at the top
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150)
    plt.close()
    
    print(f"Plot saved to: {output_filename}")


def parse_file(filename):
    """Parse ranking.txt and return (list_of_benchmark_dicts, cost_dict)."""
    with open(filename, "r") as f:
        content = f.read()

    # Strip full-line comments (lines whose first non-space char is '#')
    lines = content.split("\n")
    clean_lines = [ln for ln in lines if not ln.strip().startswith("#")]
    content = "\n".join(clean_lines)

    # Extract every top-level { … } block via brace-depth matching
    dicts = []
    i = 0
    while i < len(content):
        if content[i] == "{":
            depth = 1
            j = i + 1
            while j < len(content) and depth > 0:
                if content[j] == "{":
                    depth += 1
                elif content[j] == "}":
                    depth -= 1
                j += 1
            try:
                dicts.append(ast.literal_eval(content[i:j]))
            except (ValueError, SyntaxError) as exc:
                print(f"Warning: skipping unparseable block – {exc}",
                      file=sys.stderr)
            i = j
        else:
            i += 1

    if len(dicts) < 2:
        sys.exit("Error: need at least one benchmark dict + one cost dict.")

    # Last dict is credit-cost info; everything before it is a benchmark
    return dicts[:-1], dicts[-1]


def main():
    parser = argparse.ArgumentParser(
        description="Compute a unified model ranking from multiple benchmark leaderboards."
    )
    parser.add_argument(
        "filename",
        nargs="?",
        default="ranking.txt",
        help="Path to input file (default: ranking.txt)"
    )
    parser.add_argument(
        "-p", "--plot",
        action="store_true",
        help="Generate a PNG plot of model performance vs. cost"
    )
    args = parser.parse_args()
    
    filename = args.filename

    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.", file=sys.stderr)
        print("", file=sys.stderr)
        print("Usage: python rank_models.py [filename]", file=sys.stderr)
        print("  filename  Path to input file (default: ranking.txt)", file=sys.stderr)
        print("", file=sys.stderr)
        print("Run with 'ranking_sample.txt' for an example:", file=sys.stderr)
        print(f"  python rank_models.py ranking_sample.txt", file=sys.stderr)
        sys.exit(1)

    benchmarks, cost_dict = parse_file(filename)

    # ── Collect percentile scores per model ──────────────────────────────
    model_scores: dict[str, list[float]] = {}

    for bench in benchmarks:
        total = bench.get("known_totals")
        if not total:                       # None or 0 → skip benchmark
            continue
        for model, rank in bench.items():
            if model == "known_totals":
                continue
            if rank is None:                # model not evaluated here
                continue
            pct = rank / total
            model_scores.setdefault(model, []).append(pct)

    # ── Compute average, std-dev, apply penalty ──────────────────────────
    results = []
    for model, scores in model_scores.items():
        n = len(scores)
        if n == 0:
            continue

        mean = sum(scores) / n

        # Sparse-data penalty (added to the average)
        penalty = 0.25 if n == 1 else (0.10 if n == 2 else 0.0)
        avg = mean + penalty

        # Population std-dev of the raw percentile scores
        if n >= 2:
            var = sum((s - mean) ** 2 for s in scores) / n
            sd = math.sqrt(var)
        else:
            sd = None

        cost = cost_dict.get(model)         # None → "N/A" later
        results.append((model, avg, sd, n, cost))

    results.sort(key=lambda r: r[1])        # ascending = best first

    # ── Pretty-print ASCII table ─────────────────────────────────────────
    col = {"rank": 6, "model": 24, "avg": 10, "sd": 10, "nb": 14, "cost": 9}

    def row(*cells):
        return (f"| {cells[0]:<{col['rank']}}"
                f"| {cells[1]:<{col['model']}}"
                f"| {cells[2]:<{col['avg']}}"
                f"| {cells[3]:<{col['sd']}}"
                f"| {cells[4]:<{col['nb']}}"
                f"| {cells[5]:<{col['cost']}}|")

    width = len(row("", "", "", "", "", ""))
    sep = "+" + "-" * (width - 2) + "+"

    print(sep)
    print(row("Rank", "Model", "Avg Pctl", "Std Dev", "# Benchmarks", "Cost/1k"))
    print(sep)

    for idx, (model, avg, sd, n, cost) in enumerate(results, 1):
        avg_s = f"{avg:.3f}"
        sd_s  = f"{sd:.3f}" if sd is not None else "N/A"
        cost_s = str(cost) if cost is not None else "N/A"
        print(row(str(idx), model, avg_s, sd_s, str(n), cost_s))

    print(sep)

    # Generate plot if requested
    if args.plot:
        # Create output filename with same basename but .png extension
        base_name = os.path.splitext(filename)[0]
        plot_filename = f"{base_name}.png"
        create_plot(results, plot_filename)


if __name__ == "__main__":
    main()