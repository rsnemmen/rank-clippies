#!/usr/bin/env python
"""
Compute a unified model ranking from multiple benchmark leaderboards
using percentile-normalized scores.

Usage:  python rank_models.py [general|coding|agentic|stem]
        python rank_models.py coding --plot
"""

import argparse
import ast
import math
import os
import sys
from typing import Any

CATEGORIES: dict[str, str] = {
    "general": "General intelligence",
    "coding": "Coding and agentic coding",
    "agentic": "Agentic and computer use",
    "stem": "Math and STEM expert reasoning",
}


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
    results: list[tuple], k: float = 1.0, debug: bool = False
) -> dict[str, int]:
    """Categorize models into tiers based on IQR-based overlap with tier leaders.

    Uses the "Indistinguishable from Best" method: models whose spread intervals
    overlap with the leader's interval are placed in the same tier.

    Args:
        results: List of tuples (model, avg, spread, n, cost) from main computation
        k: Multiplier for the semi-IQR interval (1.0 = ±semi-IQR, 2.0 = wider)
        debug: If True, print detailed tiering debug output

    Returns:
        Dictionary mapping model names to tier numbers (1, 2, 3, ...)
    """
    import pandas as pd

    # Calculate average semi-IQR from models with valid spread
    valid_spreads = [spread for _, _, spread, _, _ in results if spread is not None]
    avg_spread = sum(valid_spreads) / len(valid_spreads) if valid_spreads else 0.0

    if debug:
        print("\n" + "=" * 80)
        print("TIERING DEBUG OUTPUT")
        print("=" * 80)
        print("\n📊 INITIAL DATA PREPARATION")
        print(f"   Total models: {len(results)}")
        print(f"   Models with valid semi-IQR: {len(valid_spreads)}")
        print(f"   Average semi-IQR (for models without spread): {avg_spread:.6f}")
        print(f"   k (interval multiplier): {k}")

    # Prepare data - convert scores and spreads to percentage scale
    df_data = []
    if debug:
        print("\n📋 MODEL DATA (converted to percentage scale):")
        print(f"   {'Model':<30} {'Score':<12} {'Semi-IQR':<12} {'Source':<15}")
        print(f"   {'-' * 30} {'-' * 12} {'-' * 12} {'-' * 15}")

    for model, avg, spread, n, cost in results:
        # Use average semi-IQR for models with spread=None
        effective_spread = spread if spread is not None else avg_spread
        spread_source = "measured" if spread is not None else "average"
        if debug:
            print(
                f"   {model:<30} {avg * 100:>11.4f} {effective_spread * 100:>11.4f} {spread_source:<15}"
            )
        df_data.append(
            {
                "model": model,
                "score": avg * 100,  # Convert to percentage scale
                "sd": effective_spread * 100,  # internal column name kept for DataFrame ops
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
        leader_max = leader_score + (k * leader_sd)

        if debug:
            print("\n   👑 LEADER SELECTION:")
            print(f"      Leader: {df.loc[leader_idx, 'model']}")
            print(f"      Leader score: {leader_score:.4f}")
            print(f"      Leader semi-IQR: {leader_sd:.4f}")
            print(
                f"      Leader's upper bound (score + {k}×semi-IQR): {leader_max:.4f}"
            )
            print(
                "\n   🔍 OVERLAP CHECK (candidate's lower bound ≤ leader's upper bound):"
            )
            print(f"      Leader's max (upper bound): {leader_max:.4f}")
            print(
                f"\n      {'Model':<30} {'Score':<12} {'Semi-IQR':<12} {'Lower Bound':<15} {'Overlap?':<10}"
            )
            print(f"      {'-' * 30} {'-' * 12} {'-' * 12} {'-' * 15} {'-' * 10}")

        # Find all items whose lower bound overlaps with leader's upper bound
        current_batch = df.loc[remaining_indices]
        candidate_mins = current_batch["score"] - (k * current_batch["sd"])

        # Check overlap condition: if candidate's best case overlaps with leader's worst case
        tier_members_mask = candidate_mins <= leader_max
        tier_member_indices = current_batch[tier_members_mask].index.tolist()

        if debug:
            for idx in remaining_indices:
                model = df.loc[idx, "model"]
                score = df.loc[idx, "score"]
                sd = df.loc[idx, "sd"]
                lower_bound = score - (k * sd)
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
    category: str | None = None,
    quadrants: bool = False,
    model_scores: dict[str, list[float]] | None = None,
) -> None:
    """Create a scatter plot of model performance vs. cost and save to PNG.

    Args:
        results: List of tuples (model, avg, sd, n, cost) from main computation
        output_filename: Path to save the PNG output
        debug: If True, show detailed tiering debug output
        category: Optional category label shown as subtitle
        quadrants: If True, draw quadrant dividers and labels on the plot
        model_scores: Per-model list of raw benchmark percentiles for overlay dots
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

    # Calculate average semi-IQR for models with spread=None
    valid_spreads = [spread for _, _, spread, _, _ in results if spread is not None]
    avg_spread = sum(valid_spreads) / len(valid_spreads) if valid_spreads else 0.0

    # Categorize models into tiers
    tier_mapping = categorize_tiers(results, k=1.0, debug=debug)

    # Convert results to DataFrame
    df_data = []
    for model, avg, spread, n, cost in results:
        effective_spread = spread if spread is not None else avg_spread
        df_data.append(
            {
                "Model Name": model,
                "Average Score": avg * 100,
                "Semi-IQR": effective_spread * 100,
                "Credit Cost (per 1k)": cost,
                "Tier": tier_mapping.get(model, 0),
            }
        )

    df = pd.DataFrame(df_data)
    df["Credit Cost (per 1k)"] = pd.to_numeric(df["Credit Cost (per 1k)"], errors="coerce")
    plot_df = df.dropna(subset=["Credit Cost (per 1k)"])

    # Normalize cost so best-ranked model (lowest avg score) = 1.0
    best_cost = float(plot_df.loc[plot_df["Average Score"].idxmin(), "Credit Cost (per 1k)"])
    plot_df = plot_df.copy()  # avoid pandas SettingWithCopyWarning
    plot_df["Credit Cost (per 1k)"] = plot_df["Credit Cost (per 1k)"] / best_cost

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
                    yerr=closed_data["Semi-IQR"],
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
                    yerr=open_data["Semi-IQR"],
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

        base_title = "LLM Model Performance vs. Cost"
        ax.set_title(f"{base_title}\n{category}" if category else base_title, pad=12)
        ax.set_ylabel("Percentile rank (lower = better)")
        ax.set_xlabel("Cost relative to best model (log scale)")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

        leg1 = ax.legend(loc="center right", bbox_to_anchor=(1, 0.5), title="Performance tier")
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
        ax.legend(handles=legend_elements, loc="center right", bbox_to_anchor=(1, 0.3))

        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(
                lambda x, _: f"{x:g}" if x >= 1 else f"{x:.2g}"
            )
        )
        ax.invert_yaxis()

        max_score = max(avg for _, avg, _, _, _ in results) * 100
        upper_limit = get_y_upper_limit(max_score)
        ax.set_ylim(top=0, bottom=upper_limit)

        if quadrants:
            import numpy as np

            x_mid = 10 ** float(np.mean(np.log10(plot_df["Credit Cost (per 1k)"])))
            y_mid = float(plot_df["Average Score"].median())
            x_lo, x_hi = ax.get_xlim()

            ax.axhline(y_mid, color="#aaaaaa", linewidth=0.8, linestyle="--", zorder=0)
            ax.axvline(x_mid, color="#aaaaaa", linewidth=0.8, linestyle="--", zorder=0)

            # Light background shading per quadrant
            ax.fill_between([x_lo, x_mid], [0, 0], [y_mid, y_mid], color="#a8d8a8", alpha=0.25, zorder=0)
            ax.fill_between([x_mid, x_hi], [0, 0], [y_mid, y_mid], color="#a8c8e8", alpha=0.25, zorder=0)
            ax.fill_between([x_lo, x_mid], [y_mid, y_mid], [upper_limit, upper_limit], color="#f8e8a0", alpha=0.25, zorder=0)
            ax.fill_between([x_mid, x_hi], [y_mid, y_mid], [upper_limit, upper_limit], color="#f8b8b8", alpha=0.25, zorder=0)

            label_kw = dict(fontsize=13, color="#888888", alpha=0.7, style="italic")
            ax.text(x_lo * 1.05, y_mid * 0.05, "Best value", ha="left", va="top", **label_kw)
            ax.text(x_hi * 0.95, y_mid * 0.05, "Premium", ha="right", va="top", **label_kw)
            ax.text(x_lo * 1.05, upper_limit * 0.97, "Budget", ha="left", va="bottom", **label_kw)
            ax.text(x_hi * 0.95, upper_limit * 0.97, "Avoid", ha="right", va="bottom", **label_kw)

        fig.savefig(output_filename)
        plt.close(fig)

    print(f"Plot saved to: {output_filename}")


def create_ranking_plot(
    results: list[tuple],
    output_filename: str,
    open_models: set[str] | None = None,
    debug: bool = False,
    category: str | None = None,
    model_scores: dict[str, list[float]] | None = None,
) -> None:
    """Create a horizontal ranking plot with models ordered by score.

    Args:
        results: List of tuples (model, avg, sd, n, cost) from main computation
        output_filename: Path to save the PNG output
        open_models: Set of open source model names
        debug: If True, show detailed tiering debug output
        category: Optional category label shown in the title
        model_scores: Per-model list of raw benchmark percentiles for overlay dots
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

    import random
    random.seed(42)

    valid_spreads = [spread for _, _, spread, _, _ in results if spread is not None]
    avg_spread = sum(valid_spreads) / len(valid_spreads) if valid_spreads else 0.0

    tier_mapping = categorize_tiers(results, k=1.0, debug=debug)
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

        for i, (model, avg, spread, n_bench, cost) in enumerate(sorted_results):
            effective_sd = spread if spread is not None else avg_spread
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
                zorder=2,
            )

            # Overlay individual benchmark percentiles as jittered faint dots
            if model_scores and model in model_scores:
                for pct in model_scores[model]:
                    jitter = random.gauss(0, 0.12)
                    ax.plot(
                        pct * 100,
                        i + jitter,
                        marker="o",
                        color=color,
                        alpha=0.55,
                        markersize=3,
                        markeredgecolor="white",
                        markeredgewidth=0.4,
                        zorder=4,
                    )

        y_labels = [f"{i + 1}. {model}" for i, (model, *_) in enumerate(sorted_results)]
        ax.set_yticks(range(n_models))
        ax.set_yticklabels(y_labels)

        ax.set_xlabel("Percentile rank (lower = better)")
        base_title = "Model Ranking"
        ax.set_title(f"{base_title} — {category}" if category else base_title, pad=12)
        ax.grid(True, axis="x", linestyle="--", linewidth=0.6, alpha=0.5)

        # Hide left spine; keep only the bottom axis for a clean Cleveland-dot look
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)

        ax.invert_yaxis()

        max_score = max(avg for _, avg, _, _, _ in results) * 100
        upper_limit = get_y_upper_limit(max_score)
        ax.set_xlim(left=0, right=upper_limit)

        # Tier legend (top right)
        tier_legend = [
            Line2D(
                [0], [0], marker="o", color="w",
                markerfacecolor=colors[tier_num - 1],
                markersize=9, label=f"Tier {tier_num}",
                markeredgecolor="white", markeredgewidth=0.6,
            )
            for tier_num in sorted(set(tier_mapping.values()))
        ]
        leg1 = ax.legend(handles=tier_legend, loc="upper right", title="Performance tier")
        ax.add_artist(leg1)

        # Marker-type legend: placed just below tier legend
        fig.canvas.draw()
        leg1_bb = leg1.get_window_extent()
        leg1_bottom_y = ax.transAxes.inverted().transform((0, leg1_bb.y0))[1]

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
        ax.legend(
            handles=marker_legend,
            loc="upper right",
            bbox_to_anchor=(1, leg1_bottom_y - 0.01),
            bbox_transform=ax.transAxes,
        )

        fig.savefig(output_filename)
        plt.close(fig)

    print(f"Ranking plot saved to: {output_filename}")


def _extract_raw_dicts(filepath: str) -> list[dict[str, Any]]:
    """Extract and parse all top-level dict literals from a file using brace-depth matching."""
    with open(filepath, "r") as f:
        content = f.read()
    clean = "\n".join(ln for ln in content.split("\n") if not ln.strip().startswith("#"))
    dicts: list[dict[str, Any]] = []
    i = 0
    while i < len(clean):
        if clean[i] == "{":
            depth = 1
            j = i + 1
            while j < len(clean) and depth > 0:
                if clean[j] == "{":
                    depth += 1
                elif clean[j] == "}":
                    depth -= 1
                j += 1
            try:
                dicts.append(ast.literal_eval(clean[i:j]))
            except (ValueError, SyntaxError) as exc:
                print(f"Warning: skipping unparseable block in {filepath} – {exc}", file=sys.stderr)
            i = j
        else:
            i += 1
    return dicts


def load_data(category: str) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, bool], str]:
    """Load benchmarks for a category and model metadata from centralized data files.

    Returns (benchmarks, cost_dict, open_dict, title) — same shape consumed by main().
    """
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    benchmarks_file = os.path.join(data_dir, "benchmarks.txt")
    models_file = os.path.join(data_dir, "models.txt")

    for path in (benchmarks_file, models_file):
        if not os.path.exists(path):
            sys.exit(f"Error: Required data file '{path}' not found.")

    raw_benchmarks = _extract_raw_dicts(benchmarks_file)
    raw_models = _extract_raw_dicts(models_file)

    if not raw_models:
        sys.exit("Error: No model metadata found in data/models.txt.")
    models_dict: dict[str, dict[str, Any]] = {
        k: v for k, v in raw_models[0].items() if isinstance(v, dict)
    }

    # Filter to the requested category and flatten each benchmark to the legacy schema
    benchmarks: list[dict[str, Any]] = []
    for b in raw_benchmarks:
        cats = b.get("categories")
        if not isinstance(cats, list) or category not in cats:
            continue
        scores = b.get("scores")
        if not isinstance(scores, dict):
            continue
        flat: dict[str, Any] = dict(scores)
        if "min_score" in b:
            flat["min_score"] = b["min_score"]
        if "known_totals" in b:
            flat["known_totals"] = b["known_totals"]
        benchmarks.append(flat)

    if not benchmarks:
        sys.exit(
            f"Error: No benchmarks tagged '{category}'. "
            f"Valid categories: {', '.join(CATEGORIES)}"
        )

    print(f"Loaded {len(benchmarks)} benchmarks for category '{category}'.")

    cost_dict: dict[str, float] = {
        m: float(v["cost"]) for m, v in models_dict.items() if v.get("cost") is not None
    }
    open_dict: dict[str, bool] = {
        m: True for m, v in models_dict.items() if v.get("open")
    }
    title = CATEGORIES[category]

    return benchmarks, cost_dict, open_dict, title


def main():
    parser = argparse.ArgumentParser(
        description="Compute a unified model ranking from multiple benchmark leaderboards."
    )
    parser.add_argument(
        "category",
        nargs="?",
        default="coding",
        help=f"Ranking category to compute (default: coding). Valid: {', '.join(CATEGORIES)}",
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
    parser.add_argument(
        "-q",
        "--quadrants",
        action="store_true",
        help="Add quadrant dividers and labels to the scatter plot",
    )
    args = parser.parse_args()

    category = args.category

    if category not in CATEGORIES:
        print(f"Error: Unknown category '{category}'.", file=sys.stderr)
        print(f"Valid categories: {', '.join(CATEGORIES)}", file=sys.stderr)
        sys.exit(1)

    benchmarks, cost_dict, open_dict, category = load_data(category)

    # ── Collect percentile scores per model ──────────────────────────────
    model_scores: dict[str, list[float]] = {}

    for bench in benchmarks:
        min_score = bench.get("min_score")
        if min_score is not None:
            # Score-based benchmark: higher score = better
            # Percentile: 0.0 = best, 1.0 = worst (min_score); known_totals not required
            observed = [
                v for k, v in bench.items()
                if k not in ("known_totals", "min_score") and v is not None
            ]
            if not observed:
                continue
            max_score = max(observed)
            score_range = max_score - min_score
            if score_range == 0:
                continue
            for model, score in bench.items():
                if model in ("known_totals", "min_score"):
                    continue
                if score is None:
                    continue
                pct = (max_score - score) / score_range
                model_scores.setdefault(model, []).append(pct)
        else:
            # Rank-based benchmark: lower rank number = better; known_totals required
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

        sorted_scores = sorted(scores)
        mid = n // 2
        median = sorted_scores[mid] if n % 2 == 1 else (sorted_scores[mid - 1] + sorted_scores[mid]) / 2

        # Sparse-data penalty (added to the median, capped at 1.0)
        penalty = 0.25 if n == 1 else (0.10 if n == 2 else 0.0)
        avg = min(median + penalty, 1.0)

        # Semi-IQR of raw percentile scores (robust dispersion for error bars; IQR needs n>=3)
        if n >= 3:
            q1_idx = (n - 1) * 0.25
            q3_idx = (n - 1) * 0.75
            q1 = sorted_scores[int(q1_idx)] + (q1_idx % 1) * (
                sorted_scores[min(int(q1_idx) + 1, n - 1)] - sorted_scores[int(q1_idx)]
            )
            q3 = sorted_scores[int(q3_idx)] + (q3_idx % 1) * (
                sorted_scores[min(int(q3_idx) + 1, n - 1)] - sorted_scores[int(q3_idx)]
            )
            spread = (q3 - q1) / 2
        else:
            spread = None

        cost = cost_dict.get(model)  # None → "N/A" later
        results.append((model, avg, spread, n, cost))

    results.sort(key=lambda r: r[1])  # ascending = best first

    # ── Normalize costs relative to best-ranked model with cost data ─────
    costs_with_values = [r[4] for r in results if r[4] is not None]
    best_cost_table = costs_with_values[0] if costs_with_values else None

    # ── Pretty-print ASCII table ─────────────────────────────────────────
    col = {"rank": 6, "model": 24, "avg": 10, "sd": 10, "nb": 14, "cost": 10}

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
    print(row("Rank", "Model", "Avg Pctl", "IQR/2", "# Benchmarks", "Rel. Cost"))
    print(sep)

    for idx, (model, avg, spread, n, cost) in enumerate(results, 1):
        avg_s = f"{avg:.3f}"
        sd_s = f"{spread:.3f}" if spread is not None else "N/A"
        if cost is not None and best_cost_table is not None:
            cost_s = f"{cost / best_cost_table:.3f}"
        else:
            cost_s = "N/A"
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

        stem = category
        figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
        os.makedirs(figures_dir, exist_ok=True)
        open_models = set(open_dict.keys()) if open_dict else set()

        plot_filename = os.path.join(figures_dir, f"{stem}.png")
        create_plot(plottable_results, plot_filename, open_models, debug=args.debug, category=category, quadrants=args.quadrants, model_scores=model_scores)

        ranking_plot_filename = os.path.join(figures_dir, f"{stem}_ranking.png")
        create_ranking_plot(results, ranking_plot_filename, open_models, debug=args.debug, category=category, model_scores=model_scores)
    elif args.debug:
        # Run tiering with debug output even without plot
        # Filter results to only include models with cost data for consistency
        plottable_results = [r for r in results if r[4] is not None]
        if plottable_results:
            categorize_tiers(plottable_results, k=1.0, debug=True)
        else:
            categorize_tiers(results, k=1.0, debug=True)


if __name__ == "__main__":
    main()
