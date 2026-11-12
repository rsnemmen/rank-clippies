#!/usr/bin/env python
"""
Compute a unified model ranking from multiple benchmark leaderboards.

Each benchmark score is normalized to its gap from the best tracked model
(0 = best, 1 = the benchmark's floor), a model's score is the median gap across
its benchmarks, and rank intervals and tiers come from resampling benchmarks.

Usage:  python rank_models.py [general|coding|agentic|stem]
        python rank_models.py coding --plot
"""

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tomllib
from typing import Any, NamedTuple

from ranking_math import (
    RANK_INTERVAL,
    TIER_ALPHA,
    RankingResult,
    Replicate,
    assign_tiers,
    bootstrap_scores,
    calculate_rankings,
    rank_intervals,
)

CATEGORIES: dict[str, str] = {
    "general": "General intelligence",
    "coding": "Coding and agentic coding",
    "agentic": "Agentic and computer use",
    "stem": "Math and STEM expert reasoning",
}

# Axis label shared by both plots (the website uses the same wording in docs/app.js)
_SCORE_LABEL = "Gap to best model, % of benchmark range (lower = better)"


class RawRanking(NamedTuple):
    """Ranked results plus the inputs needed to render or export them."""

    results: list[RankingResult]
    model_scores: dict[str, list[float]]
    cost_dict: dict[str, float]
    open_dict: dict[str, bool]
    company_dict: dict[str, str]
    title: str
    benchmark_names: list[str]
    replicates: list[Replicate]


# Directory containing company logo PNGs (docs/ is the GitHub Pages web root)
_LOGO_DIR = Path(__file__).parent / "docs" / "assets" / "logos"


def get_y_upper_limit(max_score: float) -> int:
    """Calculate upper y-axis limit by rounding up to next multiple of 10, capped at 100.

    Args:
        max_score: The maximum average score (in percentage scale)

    Returns:
        Upper y-axis limit as an integer, capped at 100
    """
    upper = math.ceil(max_score / 10) * 10
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

# ANSI 256-color approximations of _NATURE_COLORS (same order, for terminal output)
_OKABE_ITO_256 = [33, 214, 36, 166, 175, 117, 227, 246, 28, 89]

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


def _supports_color(no_color_flag: bool) -> bool:
    return not no_color_flag and not os.environ.get("NO_COLOR") and sys.stdout.isatty()


def _c(text: str, code: str, enabled: bool) -> str:
    return f"\x1b[{code}m{text}\x1b[0m" if (enabled and code) else text


def categorize_tiers(
    results: list[RankingResult],
    replicates: list[Replicate],
    alpha: float = TIER_ALPHA,
    debug: bool = False,
) -> dict[str, int]:
    """Group models into tiers the benchmark bootstrap cannot separate from the tier leader.

    The best remaining model leads each tier. Walking down the ranking, a model joins the
    tier while it matches or beats the leader in at least `alpha` of the resamples, so tiers
    are always contiguous in rank order. See ranking_math.assign_tiers.

    Args:
        results: Ranked tuples (model, score, lower_err, upper_err, n, cost), best first
        replicates: Bootstrap replicates from ranking_math.bootstrap_scores
        alpha: Minimum share of resamples in which a candidate must match the leader
        debug: If True, print every leader/candidate comparison

    Returns:
        Dictionary mapping model names to tier numbers (1, 2, 3, ...)
    """
    tiers, steps = assign_tiers(results, replicates, alpha)

    if debug:
        scores = {r[0]: r[1] for r in results}
        print("\n" + "=" * 80)
        print("TIERING DEBUG OUTPUT")
        print("=" * 80)
        print("\n📊 METHOD")
        print(f"   Models: {len(results)}")
        print(f"   Bootstrap replicates (benchmarks resampled): {len(replicates)}")
        print(
            f"   Join threshold: candidate at least as good as leader in >= {alpha:.0%} of resamples"
        )

        current_leader = ""
        for leader, candidate, p, joined in steps:
            if leader != current_leader:
                current_leader = leader
                print(f"\n{'─' * 80}")
                print(
                    f"TIER {tiers[leader]}  👑 leader: {leader} (score {scores[leader]:.4f})"
                )
                print(f"{'─' * 80}")
                print(
                    f"      {'Candidate':<30} {'Score':>8} {'P(as good)':>14}  Joins?"
                )
            mark = "YES ✓" if joined else "NO ✗ (leads next tier)"
            print(f"      {candidate:<30} {scores[candidate]:>8.4f} {p:>14.3f}  {mark}")

        n_tiers = max(tiers.values(), default=0)
        print("\n" + "=" * 80)
        print(f"📈 FINAL TIER ASSIGNMENTS ({n_tiers} tiers total)")
        print("=" * 80)
        for tier_num in range(1, n_tiers + 1):
            members = [r[0] for r in results if tiers[r[0]] == tier_num]
            print(f"   Tier {tier_num}: {', '.join(members)}")
        print("\n" + "=" * 80 + "\n")

    return tiers


def create_plot(
    results: list[RankingResult],
    output_filename: str,
    tier_mapping: dict[str, int],
    open_models: set[str] | None = None,
    category: str | None = None,
    quadrants: bool = False,
    model_scores: dict[str, list[float]] | None = None,
    companies: dict[str, str] | None = None,
) -> None:
    """Create a scatter plot of model performance vs. cost and save to PNG.

    Args:
        results: List of tuples (model, avg, lower_err, upper_err, n, cost) from main computation
        output_filename: Path to save the PNG output
        tier_mapping: Model name -> tier, computed over the full category ranking
        category: Optional category label shown as subtitle
        quadrants: If True, draw quadrant dividers and labels on the plot
        model_scores: Per-model list of raw benchmark percentiles for overlay dots
    """
    try:
        import matplotlib
        import pandas as pd  # type: ignore[import-untyped]
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print("Plotting requires pandas and matplotlib. Install with:", file=sys.stderr)
        print("  pip install pandas matplotlib", file=sys.stderr)
        sys.exit(1)

    # Convert results to DataFrame. Error bars show the Q1-Q3 spread across benchmarks;
    # models with n < 3 have no quartiles and get no bar.
    df_data = []
    for model, avg, lower_err, upper_err, n, cost in results:
        eff_lower = lower_err if lower_err is not None else 0.0
        eff_upper = upper_err if upper_err is not None else 0.0
        df_data.append(
            {
                "Model Name": model,
                "Average Score": avg * 100,
                "Lower": eff_lower * 100,
                "Upper": eff_upper * 100,
                "Cost (USD per 1M)": cost,
                "Tier": tier_mapping.get(model, 0),
                "N Bench": n,
            }
        )

    df = pd.DataFrame(df_data)
    df["Cost (USD per 1M)"] = pd.to_numeric(df["Cost (USD per 1M)"], errors="coerce")
    plot_df = df.dropna(subset=["Cost (USD per 1M)"])

    # Normalize cost so best-ranked model (lowest avg score) = 1.0
    best_cost = float(
        plot_df.loc[plot_df["Average Score"].idxmin(), "Cost (USD per 1M)"]
    )
    plot_df = plot_df.copy()  # avoid pandas SettingWithCopyWarning
    plot_df["Cost (USD per 1M)"] = plot_df["Cost (USD per 1M)"] / best_cost

    with matplotlib.rc_context(_NATURE_RC):
        fig, ax = plt.subplots(figsize=(11, 7))

        max_tier = int(plot_df["Tier"].max())
        colors = [_NATURE_COLORS[i % len(_NATURE_COLORS)] for i in range(max_tier)]

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
                    closed_data["Cost (USD per 1M)"].to_numpy(),
                    closed_data["Average Score"].to_numpy(),
                    yerr=[
                        closed_data["Lower"].to_numpy(),
                        closed_data["Upper"].to_numpy(),
                    ],
                    fmt="o",
                    color=color,
                    ecolor=color,
                    alpha=0.88,
                    markersize=9,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    label=f"Tier {tier_num}",
                    elinewidth=0.9,
                    capsize=2,
                    capthick=0.9,
                )

            if len(open_data) > 0:
                label = f"Tier {tier_num}" if len(closed_data) == 0 else None
                ax.errorbar(
                    open_data["Cost (USD per 1M)"].to_numpy(),
                    open_data["Average Score"].to_numpy(),
                    yerr=[open_data["Lower"].to_numpy(), open_data["Upper"].to_numpy()],
                    fmt="D",  # diamond for open-weight models
                    color=color,
                    ecolor=color,
                    alpha=0.88,
                    markersize=8,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    label=label,
                    elinewidth=0.9,
                    capsize=2,
                    capthick=0.9,
                )

        # Annotate points
        for _, row in plot_df.iterrows():
            n_b = int(row["N Bench"])
            penalty_suffix = "‡" if n_b == 1 else ("†" if n_b == 2 else "")
            ax.annotate(
                str(row["Model Name"]) + penalty_suffix,
                (float(row["Cost (USD per 1M)"]), float(row["Average Score"])),
                xytext=(5, 3),
                textcoords="offset points",
                fontsize=7,
                color="#444444",
            )

        # Overlay company logos centered on each marker
        if companies:
            logo_cache: dict[str, Any] = {}
            fig_dpi = fig.get_dpi()
            # Reference marker size in points (use the smaller diamond size for safety)
            marker_pts = 8.0
            marker_px = marker_pts * fig_dpi / 72.0

            for _, row in plot_df.iterrows():
                model_name = str(row["Model Name"])
                company = companies.get(model_name)
                if not company:
                    continue
                logo_path = _LOGO_DIR / f"{company}.png"
                if not logo_path.exists():
                    continue
                if company not in logo_cache:
                    logo_cache[company] = plt.imread(str(logo_path))
                arr = logo_cache[company]
                h, w = arr.shape[:2]
                zoom = marker_px / max(h, w) * 0.325
                imagebox = OffsetImage(arr, zoom=zoom)
                ab = AnnotationBbox(
                    imagebox,
                    (float(row["Cost (USD per 1M)"]), float(row["Average Score"])),
                    frameon=False,
                    pad=0,
                    box_alignment=(0.5, 0.5),
                    zorder=5,
                )
                ax.add_artist(ab)

        base_title = "LLM Model Performance vs. Cost"
        ax.set_title(f"{base_title}\n{category}" if category else base_title, pad=12)
        ax.set_ylabel(_SCORE_LABEL)
        ax.set_xlabel("Cost relative to best model (log scale)")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

        leg1 = ax.legend(
            loc="center right", bbox_to_anchor=(1, 0.5), title="Performance tier"
        )
        ax.add_artist(leg1)

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#666666",
                markersize=9,
                label="Proprietary",
                markeredgecolor="white",
                markeredgewidth=0.8,
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                color="w",
                markerfacecolor="#666666",
                markersize=8,
                label="Open-weight",
                markeredgecolor="white",
                markeredgewidth=0.8,
            ),
            Line2D([], [], linestyle="none", label="† few benchmarks (n=2)"),
            Line2D([], [], linestyle="none", label="‡ very few benchmarks (n=1)"),
        ]
        ax.legend(handles=legend_elements, loc="center right", bbox_to_anchor=(1, 0.3))

        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(
                lambda x, _: f"{x:g}" if x >= 1 else f"{x:.2g}"
            )
        )
        max_score = max(avg for _, avg, _, _, _, _ in results) * 100
        upper_limit = get_y_upper_limit(max_score)
        ax.set_ylim(upper_limit, 0)

        if quadrants:
            import numpy as np

            x_mid = 10 ** float(np.mean(np.log10(plot_df["Cost (USD per 1M)"])))
            y_mid = float(plot_df["Average Score"].median())
            x_lo, x_hi = ax.get_xlim()

            ax.axhline(y_mid, color="#aaaaaa", linewidth=0.8, linestyle="--", zorder=0)
            ax.axvline(x_mid, color="#aaaaaa", linewidth=0.8, linestyle="--", zorder=0)

            # Light background shading per quadrant
            ax.fill_between(
                [x_lo, x_mid],
                [0, 0],
                [y_mid, y_mid],
                color="#a8d8a8",
                alpha=0.25,
                zorder=0,
            )
            ax.fill_between(
                [x_mid, x_hi],
                [0, 0],
                [y_mid, y_mid],
                color="#a8c8e8",
                alpha=0.25,
                zorder=0,
            )
            ax.fill_between(
                [x_lo, x_mid],
                [y_mid, y_mid],
                [upper_limit, upper_limit],
                color="#f8e8a0",
                alpha=0.25,
                zorder=0,
            )
            ax.fill_between(
                [x_mid, x_hi],
                [y_mid, y_mid],
                [upper_limit, upper_limit],
                color="#f8b8b8",
                alpha=0.25,
                zorder=0,
            )

            label_kw: dict[str, Any] = {
                "fontsize": 13,
                "color": "#888888",
                "alpha": 0.7,
                "style": "italic",
            }
            ax.text(
                x_lo * 1.05, y_mid * 0.05, "Best value", ha="left", va="top", **label_kw
            )
            ax.text(
                x_hi * 0.95, y_mid * 0.05, "Premium", ha="right", va="top", **label_kw
            )
            ax.text(
                x_lo * 1.05,
                upper_limit * 0.97,
                "Budget",
                ha="left",
                va="bottom",
                **label_kw,
            )
            ax.text(
                x_hi * 0.95,
                upper_limit * 0.97,
                "Avoid",
                ha="right",
                va="bottom",
                **label_kw,
            )

        fig.savefig(output_filename)
        plt.close(fig)

    print(f"Plot saved to: {output_filename}")


def create_ranking_plot(
    results: list[RankingResult],
    output_filename: str,
    tier_mapping: dict[str, int],
    open_models: set[str] | None = None,
    category: str | None = None,
    model_scores: dict[str, list[float]] | None = None,
    intervals: dict[str, tuple[int, int]] | None = None,
) -> None:
    """Create a horizontal ranking plot with models ordered by score.

    Args:
        results: List of tuples (model, avg, lower_err, upper_err, n, cost) from main computation
        output_filename: Path to save the PNG output
        tier_mapping: Model name -> tier, computed over the full category ranking
        open_models: Set of open source model names
        category: Optional category label shown in the title
        model_scores: Per-model list of raw benchmark gaps for overlay dots
        intervals: Optional model name -> (best, worst) bootstrap rank, shown in the labels
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

    sorted_results = sorted(results, key=lambda r: r[1])
    n_models = len(sorted_results)

    with matplotlib.rc_context(_NATURE_RC):
        fig, ax = plt.subplots(figsize=(11, max(5, n_models * 0.42)))

        max_tier = max(tier_mapping.values()) if tier_mapping else 1
        colors = [_NATURE_COLORS[i % len(_NATURE_COLORS)] for i in range(max_tier)]

        costs = [cost for _, _, _, _, _, cost in sorted_results if cost is not None]
        min_cost = min(costs) if costs else 1
        max_cost = max(costs) if costs else 100
        log_min = math.log10(min_cost)
        log_max = math.log10(max_cost)

        # Alternating row backgrounds for readability
        for i in range(n_models):
            if i % 2 == 0:
                ax.axhspan(i - 0.5, i + 0.5, facecolor="#f5f5f5", alpha=1.0, zorder=0)

        for i, (model, avg, lower_err, upper_err, n_bench, cost) in enumerate(
            sorted_results
        ):
            eff_lower = lower_err if lower_err is not None else 0.0
            eff_upper = upper_err if upper_err is not None else 0.0
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
                xerr=[[eff_lower * 100], [eff_upper * 100]],
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

        def _range_suffix(model: str) -> str:
            if not intervals or model not in intervals:
                return ""
            lo, hi = intervals[model]
            return f"  [{lo}–{hi}]" if lo != hi else f"  [{lo}]"

        y_labels = [
            f"{i + 1}. {model}"
            + ("‡" if n_b == 1 else ("†" if n_b == 2 else ""))
            + _range_suffix(model)
            for i, (model, _, _, _, n_b, _) in enumerate(sorted_results)
        ]
        ax.set_yticks(range(n_models))
        ax.set_yticklabels(y_labels)

        xlabel = _SCORE_LABEL
        if intervals:
            xlabel += f"\n[a–b] = {RANK_INTERVAL:.0%} rank interval from resampling benchmarks"
        ax.set_xlabel(xlabel)
        base_title = "Model Ranking"
        ax.set_title(f"{base_title} — {category}" if category else base_title, pad=12)
        ax.grid(True, axis="x", linestyle="--", linewidth=0.6, alpha=0.5)

        # Hide left spine; keep only the bottom axis for a clean Cleveland-dot look
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)

        ax.invert_yaxis()

        max_score = max(avg for _, avg, _, _, _, _ in results) * 100
        upper_limit = get_y_upper_limit(max_score)
        ax.set_xlim(left=0, right=upper_limit)

        # Tier legend (top right)
        tier_legend = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=colors[tier_num - 1],
                markersize=9,
                label=f"Tier {tier_num}",
                markeredgecolor="white",
                markeredgewidth=0.6,
            )
            for tier_num in sorted(set(tier_mapping.values()))
        ]
        leg1 = ax.legend(
            handles=tier_legend, loc="upper right", title="Performance tier"
        )
        ax.add_artist(leg1)

        # Marker-type legend: placed just below tier legend
        fig.canvas.draw()
        leg1_bb = leg1.get_window_extent()
        leg1_bottom_y = ax.transAxes.inverted().transform((0, leg1_bb.y0))[1]

        marker_legend = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#666666",
                markersize=9,
                label="Proprietary",
                markeredgecolor="white",
                markeredgewidth=0.6,
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                color="w",
                markerfacecolor="#666666",
                markersize=8,
                label="Open-weight",
                markeredgecolor="white",
                markeredgewidth=0.6,
            ),
            Line2D([], [], linestyle="none", label="† few benchmarks (n=2)"),
            Line2D([], [], linestyle="none", label="‡ very few benchmarks (n=1)"),
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


def _load_benchmarks_toml(filepath: str) -> list[tuple[str, dict[str, Any]]]:
    """Load benchmarks from a TOML file.

    Returns (benchmark_name, benchmark_dict) pairs in file order.
    Each dict contains 'categories', 'min_score' or 'known_totals', and 'scores'.
    """
    try:
        with open(filepath, "rb") as f:
            data: dict[str, Any] = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        sys.exit(f"Error: Failed to parse TOML file '{filepath}': {exc}")
    return list(data.items())


def _load_models_toml(filepath: str) -> dict[str, dict[str, Any]]:
    """Load model metadata from a TOML file.

    Returns a dict mapping model name to its metadata dict.
    """
    with open(filepath, "rb") as f:
        return tomllib.load(f)


def load_data(
    category: str,
) -> tuple[
    list[dict[str, Any]],
    dict[str, float],
    dict[str, bool],
    dict[str, str],
    str,
    list[str],
]:
    """Load benchmarks for a category and model metadata from centralized data files.

    Returns (benchmarks, cost_dict, open_dict, company_dict, title, benchmark_names).
    company_dict maps model name to logo slug (e.g. "google", "anthropic"); models without a
    company field are omitted.
    """
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    benchmarks_file = os.path.join(data_dir, "benchmarks.toml")
    models_file = os.path.join(data_dir, "models.toml")

    for path in (benchmarks_file, models_file):
        if not os.path.exists(path):
            sys.exit(f"Error: Required data file '{path}' not found.")

    raw_benchmarks = _load_benchmarks_toml(benchmarks_file)
    models_dict = _load_models_toml(models_file)

    # Schema validation
    valid_cats = set(CATEGORIES.keys())
    for name, b in raw_benchmarks:
        cats = b.get("categories")
        if not isinstance(cats, list) or not cats:
            sys.exit(f"Error: benchmark '{name}' missing or empty 'categories' list.")
        bad_cats = set(cats) - valid_cats
        if bad_cats:
            sys.exit(
                f"Error: benchmark '{name}' has unknown categories: {bad_cats}. "
                f"Valid: {valid_cats}"
            )
        if "min_score" not in b and "known_totals" not in b:
            sys.exit(
                f"Error: benchmark '{name}' must have either 'min_score' or 'known_totals'."
            )
        scores = b.get("scores")
        if not isinstance(scores, dict):
            sys.exit(f"Error: benchmark '{name}' is missing 'scores' table.")
        # Out-of-range values silently produce normalized gaps outside 0-1
        if "min_score" in b:
            floor = b["min_score"]
            below = {m: v for m, v in scores.items() if v is not None and v < floor}
            if below:
                listed = ", ".join(f"{m}={v}" for m, v in sorted(below.items()))
                sys.exit(
                    f"Error: benchmark '{name}' has scores below min_score={floor}: {listed}. "
                    "Lower the floor or check the scores against the source."
                )
        else:
            total = b["known_totals"]
            bad = {
                m: v for m, v in scores.items() if v is not None and not 1 <= v <= total
            }
            if bad:
                listed = ", ".join(f"{m}={v}" for m, v in sorted(bad.items()))
                sys.exit(
                    f"Error: benchmark '{name}' has ranks outside 1..known_totals={total}: "
                    f"{listed}."
                )

    # Warn about model names in benchmarks.toml not present in models.toml
    all_benchmark_models: set[str] = set()
    for _, b in raw_benchmarks:
        all_benchmark_models.update(b.get("scores", {}).keys())
    orphans = all_benchmark_models - set(models_dict.keys())
    if orphans:
        print(
            f"Warning: {len(orphans)} model name(s) in benchmarks.toml not found in models.toml "
            f"(they will have no cost/open data): {', '.join(sorted(orphans))}",
            file=sys.stderr,
        )

    # Filter to the requested category and flatten each benchmark to the legacy schema
    benchmarks: list[dict[str, Any]] = []
    benchmark_names: list[str] = []
    for name, b in raw_benchmarks:
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
        flat["__name"] = name
        benchmarks.append(flat)
        benchmark_names.append(name)

    if not benchmarks:
        sys.exit(
            f"Error: No benchmarks tagged '{category}'. "
            f"Valid categories: {', '.join(CATEGORIES)}"
        )

    print(
        f"Loaded {len(benchmarks)} benchmarks for category '{category}': "
        f"{', '.join(benchmark_names)}"
    )

    cost_dict: dict[str, float] = {
        m: float(v["cost"]) for m, v in models_dict.items() if v.get("cost") is not None
    }
    open_dict: dict[str, bool] = {
        m: True for m, v in models_dict.items() if v.get("open")
    }
    company_dict: dict[str, str] = {
        m: str(v["company"]) for m, v in models_dict.items() if v.get("company")
    }
    title = CATEGORIES[category]

    return benchmarks, cost_dict, open_dict, company_dict, title, benchmark_names


def _compute_raw(category: str) -> RawRanking:
    """Load data, rank models, and bootstrap over benchmarks for rank intervals and tiers."""
    benchmarks, cost_dict, open_dict, company_dict, title, benchmark_names = load_data(
        category
    )
    results, model_scores = calculate_rankings(benchmarks, cost_dict)
    return RawRanking(
        results,
        model_scores,
        cost_dict,
        open_dict,
        company_dict,
        title,
        benchmark_names,
        bootstrap_scores(benchmarks),
    )


def compute_rankings(category: str) -> dict[str, Any]:
    """Compute rankings and return a JSON-serializable dict for web export."""
    raw = _compute_raw(category)
    results = raw.results
    tier_mapping = categorize_tiers(results, raw.replicates)
    intervals = rank_intervals(raw.replicates, results)

    costs_with_values = [r[5] for r in results if r[5] is not None]
    best_cost = costs_with_values[0] if costs_with_values else None

    models_out: list[dict[str, Any]] = []
    for idx, (model, avg, lower_err, upper_err, n, cost) in enumerate(results, 1):
        rel_cost = (
            cost / best_cost if (cost is not None and best_cost is not None) else None
        )
        rank_lo, rank_hi = intervals[model]
        models_out.append(
            {
                "rank": idx,
                "rank_lo": rank_lo,
                "rank_hi": rank_hi,
                "name": model,
                "avg_pct": round(avg, 6),
                "lower_err": round(lower_err, 6) if lower_err is not None else None,
                "upper_err": round(upper_err, 6) if upper_err is not None else None,
                "n_bench": n,
                "cost": cost,
                "rel_cost": round(rel_cost, 6) if rel_cost is not None else None,
                "tier": tier_mapping.get(model, 0),
                "is_open": model in raw.open_dict,
                "company": raw.company_dict.get(model),
                "raw_scores": [round(s, 6) for s in raw.model_scores.get(model, [])],
            }
        )

    return {
        "category": category,
        "title": raw.title,
        "generated_at": datetime.datetime.now(datetime.UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "n_benchmarks": len(raw.benchmark_names),
        "benchmark_names": raw.benchmark_names,
        "uncertainty": {
            "method": "benchmark bootstrap",
            "n_boot": len(raw.replicates),
            "rank_interval": RANK_INTERVAL,
            "tier_alpha": TIER_ALPHA,
        },
        "palette": _NATURE_COLORS,
        "models": models_out,
    }


def build_data_changelog() -> list[dict[str, str]]:
    """Return Git commits that changed benchmark or model data, newest first."""
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = [
        "git",
        "log",
        "--date=short",
        "--pretty=format:%ad%x09%s",
        "--",
        "data/benchmarks.toml",
        "data/models.toml",
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    entries: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        date, _, message = line.partition("\t")
        if date and message:
            entries.append({"date": date, "message": message})
    return entries


def main() -> None:
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
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color in terminal output",
    )
    parser.add_argument(
        "--export-json",
        nargs="?",
        const="docs/data",
        default=None,
        metavar="DIR",
        help="Export ranking data as JSON for all categories to DIR (default: docs/data). Requires pandas.",
    )
    args = parser.parse_args()
    color = _supports_color(args.no_color)

    if args.export_json is not None:
        out_dir = args.export_json
        os.makedirs(out_dir, exist_ok=True)
        for cat in CATEGORIES:
            data = compute_rankings(cat)
            out_path = os.path.join(out_dir, f"{cat}.json")
            with open(out_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Exported: {out_path}")
        changelog_path = os.path.join(out_dir, "changelog.json")
        with open(changelog_path, "w") as f:
            json.dump({"entries": build_data_changelog()}, f, indent=2)
        print(f"Exported: {changelog_path}")
        return

    category = args.category

    if category not in CATEGORIES:
        print(f"Error: Unknown category '{category}'.", file=sys.stderr)
        print(f"Valid categories: {', '.join(CATEGORIES)}", file=sys.stderr)
        sys.exit(1)

    raw = _compute_raw(category)
    results = raw.results
    open_dict = raw.open_dict
    title = raw.title
    tier_mapping = categorize_tiers(results, raw.replicates)
    intervals = rank_intervals(raw.replicates, results)

    # ── Normalize costs relative to best-ranked model with cost data ─────
    costs_with_values = [r[5] for r in results if r[5] is not None]
    best_cost_table = costs_with_values[0] if costs_with_values else None

    # ── Pretty-print ASCII table ─────────────────────────────────────────
    col = {
        "rank": 6,
        "model": 24,
        "range": 11,
        "avg": 7,
        "lower": 7,
        "upper": 7,
        "nb": 14,
        "cost": 10,
    }
    total_w = sum(col.values()) + 2 * len(col) + 1
    sep = "+" + "-" * (total_w - 2) + "+"

    def _row(*cells: str) -> str:
        return "".join(f"| {cell}" for cell in cells) + "|"

    sep_line = _c(sep, "2", color)
    hdr = (
        _c(f"{'Rank':<{col['rank']}}", "1", color),
        _c(f"{'Model':<{col['model']}}", "1", color),
        _c(f"{'Rank range':<{col['range']}}", "1", color),
        _c(f"{'Gap':<{col['avg']}}", "1", color),
        _c(f"{'-IQR':<{col['lower']}}", "1", color),
        _c(f"{'+IQR':<{col['upper']}}", "1", color),
        _c(f"{'# Benchmarks':<{col['nb']}}", "1", color),
        _c(f"{'Rel. Cost':<{col['cost']}}", "1", color),
    )

    print(sep_line)
    print(_row(*hdr))
    print(sep_line)

    _rank_codes = {1: "1;38;5;220", 2: "1;38;5;250", 3: "1;38;5;208"}

    for idx, (model, avg, lower_err, upper_err, n, cost) in enumerate(results, 1):
        rank_lo, rank_hi = intervals[model]
        range_s = f"{rank_lo}-{rank_hi}" if rank_lo != rank_hi else f"{rank_lo}"
        avg_s = f"{avg:.3f}"
        lower_s = f"{lower_err:.3f}" if lower_err is not None else "N/A"
        upper_s = f"{upper_err:.3f}" if upper_err is not None else "N/A"
        cost_val = (
            cost / best_cost_table
            if (cost is not None and best_cost_table is not None)
            else None
        )
        cost_s = f"{cost_val:.3f}" if cost_val is not None else "N/A"

        model_disp = model + ("*" if model in open_dict else "")

        # Pad to column width before applying color so ANSI codes don't affect alignment
        rank_p = _c(f"{idx!s:<{col['rank']}}", _rank_codes.get(idx, ""), color)
        model_p = f"{model_disp:<{col['model']}}"
        range_p = f"{range_s:<{col['range']}}"
        avg_p = f"{avg_s:<{col['avg']}}"
        lower_p = f"{lower_s:<{col['lower']}}"
        upper_p = f"{upper_s:<{col['upper']}}"
        nb_p = f"{n!s:<{col['nb']}}"
        cost_p = f"{cost_s:<{col['cost']}}"

        tier = tier_mapping.get(model, 0)
        if 1 <= tier <= len(_OKABE_ITO_256):
            model_p = _c(model_p, f"38;5;{_OKABE_ITO_256[tier - 1]}", color)

        if cost_val is not None:
            cost_code = (
                "32" if cost_val <= 2.0 else ("33" if cost_val <= 10.0 else "31")
            )
            cost_p = _c(cost_p, cost_code, color)

        print(_row(rank_p, model_p, range_p, avg_p, lower_p, upper_p, nb_p, cost_p))

    print(sep_line)
    print(
        "  Gap: median over benchmarks of (best - score) / (best - floor); 0 = best tracked model."
    )
    print(
        f"  Rank range: {RANK_INTERVAL:.0%} interval from resampling benchmarks. "
        "-IQR/+IQR: spread across benchmarks."
    )
    if color and open_dict:
        print("  * open-weight model")

    # Generate plot if requested
    if args.plot:
        # Filter results to only include models with cost data
        plottable_results = [r for r in results if r[5] is not None]
        excluded = [r[0] for r in results if r[5] is None]

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

        figures_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "figures"
        )
        os.makedirs(figures_dir, exist_ok=True)
        open_models = set(open_dict.keys()) if open_dict else set()

        plot_filename = os.path.join(figures_dir, f"{title}.png")
        create_plot(
            plottable_results,
            plot_filename,
            tier_mapping,
            open_models,
            category=title,
            quadrants=args.quadrants,
            model_scores=raw.model_scores,
            companies=raw.company_dict,
        )

        ranking_plot_filename = os.path.join(figures_dir, f"{title}_ranking.png")
        create_ranking_plot(
            results,
            ranking_plot_filename,
            tier_mapping,
            open_models,
            category=title,
            model_scores=raw.model_scores,
            intervals=intervals,
        )

    if args.debug:
        # Tiers cover every ranked model, including those without cost data
        categorize_tiers(results, raw.replicates, debug=True)


if __name__ == "__main__":
    main()
