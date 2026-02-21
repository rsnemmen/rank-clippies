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


def get_y_upper_limit(max_score: float) -> int:
    """Calculate upper y-axis limit by rounding up to next multiple of 10, capped at 100.

    Args:
        max_score: The maximum average score (in percentage scale)

    Returns:
        Upper y-axis limit as an integer, capped at 100
    """
    upper = (int(max_score) // 10 + 1) * 10
    return min(upper, 100)


# Okabe-Ito colorblind-safe palette (recommended by Nature journals)
_NATURE_COLORS = [
    "#0072B2",  # deep blue      (Tier 1 – best)
    "#E69F00",  # orange
    "#009E73",  # teal green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#999999",  # medium gray
    "#117733",  # dark green
    "#882255",  # wine
]

# rcParams for a Nature-style figure
_NATURE_RC: dict[str, object] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.labelsize": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "grid.color": "#d0d0d0",
    "grid.linewidth": 0.6,
    "legend.fontsize": 9,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#cccccc",
    "legend.title_fontsize": 9,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def categorize_tiers(
    results: list[tuple], z_score: float = 1.0, debug: bool = False
) -> dict[str, int]:
    """Categorize models into tiers based on statistical overlap with tier leaders.

    Uses the "Indistinguishable from Best" method: models whose error bars overlap
    with the leader's error bar are placed in the same tier.

    Args:
        results: List of tuples (model, avg, sd, n, cost) from main computation
        z_score: Confidence level (1.0 = 68% CI, 1.96 = 95% CI)
        debug: If True, print detailed tiering debug output

    Returns:
        Dictionary mapping model names to tier numbers (1, 2, 3, ...)
    """
    import pandas as pd

    # Calculate average SD from models with valid SD
    valid_sds = [sd for _, _, sd, _, _ in results if sd is not None]
    avg_sd = sum(valid_sds) / len(valid_sds) if valid_sds else 0.0

    if debug:
        print("\n" + "=" * 80)
        print("TIERING DEBUG OUTPUT")
        print("=" * 80)
        print("\n📊 INITIAL DATA PREPARATION")
        print(f"   Total models: {len(results)}")
        print(f"   Models with valid SD: {len(valid_sds)}")
        print(f"   Average SD (for models without SD): {avg_sd:.6f}")
        print(f"   Z-score (confidence level): {z_score}")

    # Prepare data - convert scores and SDs to percentage scale
    df_data = []
    if debug:
        print("\n📋 MODEL DATA (converted to percentage scale):")
        print(f"   {'Model':<30} {'Score':<12} {'SD':<12} {'SD Source':<15}")
        print(f"   {'-' * 30} {'-' * 12} {'-' * 12} {'-' * 15}")

    for model, avg, sd, n, cost in results:
        # Use average SD for models with sd=None
        effective_sd = sd if sd is not None else avg_sd
        sd_source = "measured" if sd is not None else "average"
        if debug:
            print(
                f"   {model:<30} {avg * 100:>11.4f} {effective_sd * 100:>11.4f} {sd_source:<15}"
            )
        df_data.append(
            {
                "model": model,
                "score": avg * 100,  # Convert to percentage scale
                "sd": effective_sd * 100,  # Convert to percentage scale
            }
        )

    df = pd.DataFrame(df_data)

    # Sort by score (ascending) - lowest scores (best performers) first
    df = df.sort_values(by="score", ascending=True).reset_index(drop=True)
    df["tier"] = 0

    if debug:
        print("\n🔄 SORTING: Models sorted by score (ascending - best first)")
        print(
            f"   Best performer: {df.loc[0, 'model']} (score: {df.loc[0, 'score']:.4f})"
        )
        print(
            f"   Worst performer: {df.loc[len(df) - 1, 'model']} (score: {df.loc[len(df) - 1, 'score']:.4f})"
        )
        print("\n" + "=" * 80)
        print("BEGINNING TIER ASSIGNMENT LOOP")
        print("=" * 80)

    current_tier = 1
    remaining_indices = df.index.tolist()

    while len(remaining_indices) > 0:
        if debug:
            print(f"\n{'─' * 80}")
            print(f"TIER {current_tier} ASSIGNMENT")
            print(f"{'─' * 80}")
            print(f"   Remaining models to tier: {len(remaining_indices)}")
            print(
                f"   Remaining model names: {[df.loc[i, 'model'] for i in remaining_indices]}"
            )

        # Identify the leader (lowest scoring/best performing remaining item)
        leader_idx = remaining_indices[0]
        leader_score = df.loc[leader_idx, "score"]
        leader_sd = df.loc[leader_idx, "sd"]
        leader_max = leader_score + (z_score * leader_sd)

        if debug:
            print("\n   👑 LEADER SELECTION:")
            print(f"      Leader: {df.loc[leader_idx, 'model']}")
            print(f"      Leader score: {leader_score:.4f}")
            print(f"      Leader SD: {leader_sd:.4f}")
            print(
                f"      Leader's upper bound (score + {z_score}×SD): {leader_max:.4f}"
            )
            print(
                "\n   🔍 OVERLAP CHECK (candidate's lower bound ≤ leader's upper bound):"
            )
            print(f"      Leader's max (upper bound): {leader_max:.4f}")
            print(
                f"\n      {'Model':<30} {'Score':<12} {'SD':<12} {'Lower Bound':<15} {'Overlap?':<10}"
            )
            print(f"      {'-' * 30} {'-' * 12} {'-' * 12} {'-' * 15} {'-' * 10}")

        # Find all items whose lower bound overlaps with leader's upper bound
        current_batch = df.loc[remaining_indices]
        candidate_mins = current_batch["score"] - (z_score * current_batch["sd"])

        # Check overlap condition: if candidate's best case overlaps with leader's worst case
        tier_members_mask = candidate_mins <= leader_max
        tier_member_indices = current_batch[tier_members_mask].index.tolist()

        if debug:
            for idx in remaining_indices:
                model = df.loc[idx, "model"]
                score = df.loc[idx, "score"]
                sd = df.loc[idx, "sd"]
                lower_bound = score - (z_score * sd)
                overlaps = lower_bound <= leader_max
                overlap_str = "YES ✓" if overlaps else "NO ✗"
                print(
                    f"      {model:<30} {score:>11.4f} {sd:>11.4f} {lower_bound:>14.4f} {overlap_str:<10}"
                )

        # Assign tier
        if debug:
            print(f"\n   ✅ TIER {current_tier} MEMBERS:")
            for idx in tier_member_indices:
                model = df.loc[idx, "model"]
                score = df.loc[idx, "score"]
                print(f"      • {model} (score: {score:.4f})")

        df.loc[tier_member_indices, "tier"] = current_tier

        # Remove tiered items from remaining list
        remaining_indices = [
            idx for idx in remaining_indices if idx not in tier_member_indices
        ]

        if debug:
            print("\n   📊 PROGRESS:")
            print(
                f"      Assigned {len(tier_member_indices)} model(s) to Tier {current_tier}"
            )
            print(f"      Remaining models: {len(remaining_indices)}")
            if len(remaining_indices) > 0:
                print(
                    f"      Next leader will be: {df.loc[remaining_indices[0], 'model']}"
                )
            else:
                print("      All models have been assigned to tiers!")

        current_tier += 1

    # Final summary
    if debug:
        total_tiers = current_tier - 1
        print("\n" + "=" * 80)
        print("TIERING COMPLETE")
        print("=" * 80)
        print(f"\n📈 FINAL TIER ASSIGNMENTS ({total_tiers} tiers total):")

        for tier_num in range(1, total_tiers + 1):
            tier_models = df[df["tier"] == tier_num]["model"].tolist()
            print(f"\n   Tier {tier_num}: {len(tier_models)} model(s)")
            for model in tier_models:
                score = df[df["model"] == model]["score"].values[0]
                print(f"      • {model} (score: {score:.4f})")

        print("\n" + "=" * 80 + "\n")

    # Create mapping from model name to tier
    return dict(zip(df["model"], df["tier"]))


def create_plot(
    results: list[tuple],
    output_filename: str,
    open_models: set[str] | None = None,
    debug: bool = False,
) -> None:
    """Create a scatter plot of model performance vs. cost and save to PNG.

    Args:
        results: List of tuples (model, avg, sd, n, cost) from main computation
        output_filename: Path to save the PNG output
        debug: If True, show detailed tiering debug output
    """
    try:
        import matplotlib
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print("Plotting requires pandas and matplotlib. Install with:", file=sys.stderr)
        print("  pip install pandas matplotlib", file=sys.stderr)
        sys.exit(1)

    # Calculate average SD for models with sd=None
    valid_sds = [sd for _, _, sd, _, _ in results if sd is not None]
    avg_sd = sum(valid_sds) / len(valid_sds) if valid_sds else 0.0

    # Categorize models into tiers
    tier_mapping = categorize_tiers(results, z_score=1.0, debug=debug)

    # Convert results to DataFrame
    df_data = []
    for model, avg, sd, n, cost in results:
        effective_sd = sd if sd is not None else avg_sd
        df_data.append(
            {
                "Model Name": model,
                "Average Score": avg * 100,
                "Std Dev": effective_sd * 100,
                "Credit Cost (per 1k)": cost,
                "Tier": tier_mapping.get(model, 0),
            }
        )

    df = pd.DataFrame(df_data)
    df["Credit Cost (per 1k)"] = pd.to_numeric(df["Credit Cost (per 1k)"], errors="coerce")
    plot_df = df.dropna(subset=["Credit Cost (per 1k)"])

    with matplotlib.rc_context(_NATURE_RC):
        fig, ax = plt.subplots(figsize=(11, 7))

        max_tier = int(plot_df["Tier"].max())
        colors = [_NATURE_COLORS[i % len(_NATURE_COLORS)] for i in range(max_tier)]

        eb_kw = dict(elinewidth=0.9, capsize=2, capthick=0.9)

        for tier_num in sorted(plot_df["Tier"].unique()):
            tier_data = plot_df[plot_df["Tier"] == tier_num]
            color = colors[tier_num - 1]

            open_data = (
                tier_data[tier_data["Model Name"].isin(open_models)]
                if open_models
                else pd.DataFrame()
            )
            closed_data = (
                tier_data[~tier_data["Model Name"].isin(open_models)]
                if open_models
                else tier_data
            )

            if len(closed_data) > 0:
                ax.errorbar(
                    closed_data["Credit Cost (per 1k)"],
                    closed_data["Average Score"],
                    yerr=closed_data["Std Dev"],
                    fmt="o",
                    color=color,
                    ecolor=color,
                    alpha=0.88,
                    markersize=9,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    label=f"Tier {tier_num}",
                    **eb_kw,
                )

            if len(open_data) > 0:
                label = f"Tier {tier_num}" if len(closed_data) == 0 else None
                ax.errorbar(
                    open_data["Credit Cost (per 1k)"],
                    open_data["Average Score"],
                    yerr=open_data["Std Dev"],
                    fmt="D",  # diamond for open-weight models
                    color=color,
                    ecolor=color,
                    alpha=0.88,
                    markersize=8,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    label=label,
                    **eb_kw,
                )

        # Annotate points
        for _, row in plot_df.iterrows():
            ax.annotate(
                str(row["Model Name"]),
                (float(row["Credit Cost (per 1k)"]), float(row["Average Score"])),
                xytext=(5, 3),
                textcoords="offset points",
                fontsize=7,
                color="#444444",
            )

        ax.set_title("LLM Model Performance vs. Cost", pad=12)
        ax.set_ylabel("Percentile rank (lower = better)")
        ax.set_xlabel("Credit cost per 1 k tokens (log scale)")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

        leg1 = ax.legend(loc="best", title="Performance tier")
        ax.add_artist(leg1)

        legend_elements = [
            Line2D(
                [0], [0], marker="o", color="w", markerfacecolor="#666666",
                markersize=9, label="Proprietary", markeredgecolor="white", markeredgewidth=0.8,
            ),
            Line2D(
                [0], [0], marker="D", color="w", markerfacecolor="#666666",
                markersize=8, label="Open-weight", markeredgecolor="white", markeredgewidth=0.8,
            ),
        ]
        ax.legend(handles=legend_elements, loc="lower right")

        ax.set_xscale("log")
        ax.invert_yaxis()

        max_score = max(avg for _, avg, _, _, _ in results) * 100
        upper_limit = get_y_upper_limit(max_score)
        ax.set_ylim(top=0, bottom=upper_limit)

        fig.savefig(output_filename)
        plt.close(fig)

    print(f"Plot saved to: {output_filename}")


def create_ranking_plot(
    results: list[tuple],
    output_filename: str,
    open_models: set[str] | None = None,
    debug: bool = False,
) -> None:
    """Create a horizontal ranking plot with models ordered by score.

    Args:
        results: List of tuples (model, avg, sd, n, cost) from main computation
        output_filename: Path to save the PNG output
        open_models: Set of open source model names
        debug: If True, show detailed tiering debug output
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print("Plotting requires matplotlib. Install with:", file=sys.stderr)
        print("  pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    valid_sds = [sd for _, _, sd, _, _ in results if sd is not None]
    avg_sd = sum(valid_sds) / len(valid_sds) if valid_sds else 0.0

    tier_mapping = categorize_tiers(results, z_score=1.0, debug=debug)
    sorted_results = sorted(results, key=lambda r: r[1])
    n_models = len(sorted_results)

    with matplotlib.rc_context(_NATURE_RC):
        fig, ax = plt.subplots(figsize=(11, max(5, n_models * 0.42)))

        max_tier = max(tier_mapping.values()) if tier_mapping else 1
        colors = [_NATURE_COLORS[i % len(_NATURE_COLORS)] for i in range(max_tier)]

        costs = [cost for _, _, _, _, cost in sorted_results if cost is not None]
        min_cost = min(costs) if costs else 1
        max_cost = max(costs) if costs else 100
        log_min = math.log10(min_cost)
        log_max = math.log10(max_cost)

        # Alternating row backgrounds for readability
        for i in range(n_models):
            if i % 2 == 0:
                ax.axhspan(i - 0.5, i + 0.5, facecolor="#f5f5f5", alpha=1.0, zorder=0)

        for i, (model, avg, sd, n_bench, cost) in enumerate(sorted_results):
            effective_sd = sd if sd is not None else avg_sd
            tier = tier_mapping.get(model, 1)
            color = colors[tier - 1]
            is_open = open_models and model in open_models
            marker = "D" if is_open else "o"

            if cost is not None and log_max != log_min:
                normalized = (math.log10(cost) - log_min) / (log_max - log_min)
                markersize = 5 + normalized * 12
            else:
                markersize = 7

            ax.errorbar(
                avg * 100,
                i,
                xerr=effective_sd * 100,
                fmt=marker,
                color=color,
                ecolor=color,
                elinewidth=0.9,
                alpha=0.88,
                markersize=markersize,
                capsize=2,
                capthick=0.9,
                markeredgecolor="white",
                markeredgewidth=0.6,
                zorder=3,
            )

        y_labels = [f"{i + 1}. {model}" for i, (model, *_) in enumerate(sorted_results)]
        ax.set_yticks(range(n_models))
        ax.set_yticklabels(y_labels)

        ax.set_xlabel("Percentile rank (lower = better)")
        ax.set_title("Model Ranking", pad=12)
        ax.grid(True, axis="x", linestyle="--", linewidth=0.6, alpha=0.5)

        # Hide left spine; keep only the bottom axis for a clean Cleveland-dot look
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)

        ax.invert_yaxis()

        max_score = max(avg for _, avg, _, _, _ in results) * 100
        upper_limit = get_y_upper_limit(max_score)
        ax.set_xlim(left=0, right=upper_limit)

        # Tier legend
        tier_legend = [
            Line2D(
                [0], [0], marker="o", color="w",
                markerfacecolor=colors[tier_num - 1],
                markersize=9, label=f"Tier {tier_num}",
                markeredgecolor="white", markeredgewidth=0.6,
            )
            for tier_num in sorted(set(tier_mapping.values()))
        ]
        leg1 = ax.legend(handles=tier_legend, loc="lower right", title="Performance tier")
        ax.add_artist(leg1)

        # Marker-type legend
        marker_legend = [
            Line2D(
                [0], [0], marker="o", color="w", markerfacecolor="#666666",
                markersize=9, label="Proprietary", markeredgecolor="white", markeredgewidth=0.6,
            ),
            Line2D(
                [0], [0], marker="D", color="w", markerfacecolor="#666666",
                markersize=8, label="Open-weight", markeredgecolor="white", markeredgewidth=0.6,
            ),
        ]
        ax.legend(handles=marker_legend, loc="upper right")

        fig.savefig(output_filename)
        plt.close(fig)

    print(f"Ranking plot saved to: {output_filename}")


def parse_file(filename):
    """Parse ranking.txt and return (list_of_benchmark_dicts, cost_dict, open_dict)."""
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
                print(f"Warning: skipping unparseable block – {exc}", file=sys.stderr)
            i = j
        else:
            i += 1

    if len(dicts) < 2:
        sys.exit("Error: need at least one benchmark dict + one cost dict.")

    # Find the 'open' dictionary (if exists) and 'cost' dictionary (last one)
    cost_dict = dicts[-1]
    open_dict = {}

    # Find open dict: has no "known_totals" (not a benchmark) and no "cost" key
    for d in dicts[:-1]:
        if "known_totals" not in d and "cost" not in d:
            open_dict = d
            break

    # Benchmarks are all dicts with "known_totals"
    benchmarks = [d for d in dicts[:-1] if "known_totals" in d]

    return benchmarks, cost_dict, open_dict


def main():
    parser = argparse.ArgumentParser(
        description="Compute a unified model ranking from multiple benchmark leaderboards."
    )
    parser.add_argument(
        "filename",
        nargs="?",
        default="ranking.txt",
        help="Path to input file (default: ranking.txt)",
    )
    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="Generate a PNG plot of model performance vs. cost",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Show detailed tiering debug output",
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
        print("  python rank_models.py ranking_sample.txt", file=sys.stderr)
        sys.exit(1)

    benchmarks, cost_dict, open_dict = parse_file(filename)

    # ── Collect percentile scores per model ──────────────────────────────
    model_scores: dict[str, list[float]] = {}

    for bench in benchmarks:
        total = bench.get("known_totals")
        if not total:  # None or 0 → skip benchmark
            continue
        for model, rank in bench.items():
            if model == "known_totals":
                continue
            if rank is None:  # model not evaluated here
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

        # Sparse-data penalty (added to the average, capped at 1.0)
        penalty = 0.25 if n == 1 else (0.10 if n == 2 else 0.0)
        avg = min(mean + penalty, 1.0)

        # Population std-dev of the raw percentile scores
        if n >= 2:
            var = sum((s - mean) ** 2 for s in scores) / n
            sd = math.sqrt(var)
        else:
            sd = None

        cost = cost_dict.get(model)  # None → "N/A" later
        results.append((model, avg, sd, n, cost))

    results.sort(key=lambda r: r[1])  # ascending = best first

    # ── Pretty-print ASCII table ─────────────────────────────────────────
    col = {"rank": 6, "model": 24, "avg": 10, "sd": 10, "nb": 14, "cost": 9}

    def row(*cells):
        return (
            f"| {cells[0]:<{col['rank']}}"
            f"| {cells[1]:<{col['model']}}"
            f"| {cells[2]:<{col['avg']}}"
            f"| {cells[3]:<{col['sd']}}"
            f"| {cells[4]:<{col['nb']}}"
            f"| {cells[5]:<{col['cost']}}|"
        )

    width = len(row("", "", "", "", "", ""))
    sep = "+" + "-" * (width - 2) + "+"

    print(sep)
    print(row("Rank", "Model", "Avg Pctl", "Std Dev", "# Benchmarks", "Cost/1k"))
    print(sep)

    for idx, (model, avg, sd, n, cost) in enumerate(results, 1):
        avg_s = f"{avg:.3f}"
        sd_s = f"{sd:.3f}" if sd is not None else "N/A"
        cost_s = str(cost) if cost is not None else "N/A"
        print(row(str(idx), model, avg_s, sd_s, str(n), cost_s))

    print(sep)

    # Generate plot if requested
    if args.plot:
        # Filter results to only include models with cost data
        plottable_results = [r for r in results if r[4] is not None]
        excluded = [r[0] for r in results if r[4] is None]

        if excluded:
            print(
                f"\nNote: {len(excluded)} model(s) excluded from cost plot (no cost data): {', '.join(excluded)}"
            )

        if not plottable_results:
            print(
                "\nError: No models with cost data available for plotting.",
                file=sys.stderr,
            )
            sys.exit(1)

        base_name = os.path.splitext(filename)[0]
        open_models = set(open_dict.keys()) if open_dict else set()

        plot_filename = f"{base_name}.png"
        create_plot(plottable_results, plot_filename, open_models, debug=args.debug)

        ranking_plot_filename = f"{base_name}_ranking.png"
        create_ranking_plot(results, ranking_plot_filename, open_models, debug=args.debug)
    elif args.debug:
        # Run tiering with debug output even without plot
        # Filter results to only include models with cost data for consistency
        plottable_results = [r for r in results if r[4] is not None]
        if plottable_results:
            categorize_tiers(plottable_results, z_score=1.0, debug=True)
        else:
            categorize_tiers(results, z_score=1.0, debug=True)


if __name__ == "__main__":
    main()
