#!/usr/bin/env python3
"""
Compute a unified model ranking from multiple benchmark leaderboards
using percentile-normalized scores.

Usage:  python rank_models.py [ranking.txt]
"""

import ast
import math
import sys


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
    filename = sys.argv[1] if len(sys.argv) > 1 else "ranking.txt"
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


if __name__ == "__main__":
    main()