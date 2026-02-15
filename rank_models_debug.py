#!/usr/bin/env python
"""
Debug version of rank_models.py - prints tiering process details.

Usage:  python rank_models_debug.py [ranking.txt]
        python rank_models_debug.py [ranking.txt] --plot
"""

import argparse
import ast
import math
import os
import sys


def get_y_upper_limit(max_score: float) -> int:
    """Calculate upper y-axis limit by rounding up to next multiple of 10, capped at 100."""
    upper = (int(max_score) // 10 + 1) * 10
    return min(upper, 100)


def categorize_tiers_debug(results: list[tuple], z_score: float = 1.0) -> dict[str, int]:
    """Categorize models into tiers with detailed debug output."""
    import pandas as pd
    
    print("\n" + "="*80)
    print("TIERING DEBUG OUTPUT")
    print("="*80)
    
    # Calculate average SD from models with valid SD
    valid_sds = [sd for _, _, sd, _, _ in results if sd is not None]
    avg_sd = sum(valid_sds) / len(valid_sds) if valid_sds else 0.0
    
    print(f"\n📊 INITIAL DATA PREPARATION")
    print(f"   Total models: {len(results)}")
    print(f"   Models with valid SD: {len(valid_sds)}")
    print(f"   Average SD (for models without SD): {avg_sd:.6f}")
    print(f"   Z-score (confidence level): {z_score}")
    
    # Prepare data - convert scores and SDs to percentage scale
    df_data = []
    print(f"\n📋 MODEL DATA (converted to percentage scale):")
    print(f"   {'Model':<30} {'Score':<12} {'SD':<12} {'SD Source':<15}")
    print(f"   {'-'*30} {'-'*12} {'-'*12} {'-'*15}")
    
    for model, avg, sd, n, cost in results:
        effective_sd = sd if sd is not None else avg_sd
        sd_source = "measured" if sd is not None else "average"
        print(f"   {model:<30} {avg*100:>11.4f} {effective_sd*100:>11.4f} {sd_source:<15}")
        df_data.append({
            'model': model,
            'score': avg * 100,
            'sd': effective_sd * 100,
        })
    
    df = pd.DataFrame(df_data)
    
    # Sort by score (ascending) - lowest scores (best performers) first
    df = df.sort_values(by='score', ascending=True).reset_index(drop=True)
    df['tier'] = 0
    
    print(f"\n🔄 SORTING: Models sorted by score (ascending - best first)")
    print(f"   Best performer: {df.loc[0, 'model']} (score: {df.loc[0, 'score']:.4f})")
    print(f"   Worst performer: {df.loc[len(df)-1, 'model']} (score: {df.loc[len(df)-1, 'score']:.4f})")
    
    current_tier = 1
    remaining_indices = df.index.tolist()
    
    print(f"\n" + "="*80)
    print("BEGINNING TIER ASSIGNMENT LOOP")
    print("="*80)
    
    while len(remaining_indices) > 0:
        print(f"\n{'─'*80}")
        print(f"TIER {current_tier} ASSIGNMENT")
        print(f"{'─'*80}")
        print(f"   Remaining models to tier: {len(remaining_indices)}")
        print(f"   Remaining model names: {[df.loc[i, 'model'] for i in remaining_indices]}")
        
        # Identify the leader (lowest scoring/best performing remaining item)
        leader_idx = remaining_indices[0]
        leader_score = df.loc[leader_idx, 'score']
        leader_sd = df.loc[leader_idx, 'sd']
        leader_max = leader_score + (z_score * leader_sd)
        
        print(f"\n   👑 LEADER SELECTION:")
        print(f"      Leader: {df.loc[leader_idx, 'model']}")
        print(f"      Leader score: {leader_score:.4f}")
        print(f"      Leader SD: {leader_sd:.4f}")
        print(f"      Leader's upper bound (score + {z_score}×SD): {leader_max:.4f}")
        
        # Find all items whose lower bound overlaps with leader's upper bound
        current_batch = df.loc[remaining_indices]
        
        print(f"\n   🔍 OVERLAP CHECK (candidate's lower bound ≤ leader's upper bound):")
        print(f"      Leader's max (upper bound): {leader_max:.4f}")
        print(f"\n      {'Model':<30} {'Score':<12} {'SD':<12} {'Lower Bound':<15} {'Overlap?':<10}")
        print(f"      {'-'*30} {'-'*12} {'-'*12} {'-'*15} {'-'*10}")
        
        tier_member_indices = []
        for idx in remaining_indices:
            model = df.loc[idx, 'model']
            score = df.loc[idx, 'score']
            sd = df.loc[idx, 'sd']
            lower_bound = score - (z_score * sd)
            overlaps = lower_bound <= leader_max
            overlap_str = "YES ✓" if overlaps else "NO ✗"
            
            print(f"      {model:<30} {score:>11.4f} {sd:>11.4f} {lower_bound:>14.4f} {overlap_str:<10}")
            
            if overlaps:
                tier_member_indices.append(idx)
        
        # Assign tier
        print(f"\n   ✅ TIER {current_tier} MEMBERS:")
        for idx in tier_member_indices:
            model = df.loc[idx, 'model']
            score = df.loc[idx, 'score']
            print(f"      • {model} (score: {score:.4f})")
            df.loc[idx, 'tier'] = current_tier
        
        # Remove tiered items from remaining list
        remaining_indices = [idx for idx in remaining_indices if idx not in tier_member_indices]
        
        print(f"\n   📊 PROGRESS:")
        print(f"      Assigned {len(tier_member_indices)} model(s) to Tier {current_tier}")
        print(f"      Remaining models: {len(remaining_indices)}")
        if len(remaining_indices) > 0:
            print(f"      Next leader will be: {df.loc[remaining_indices[0], 'model']}")
        else:
            print(f"      All models have been assigned to tiers!")
        
        current_tier += 1
    
    # Final summary
    total_tiers = current_tier - 1
    print(f"\n" + "="*80)
    print("TIERING COMPLETE")
    print("="*80)
    print(f"\n📈 FINAL TIER ASSIGNMENTS ({total_tiers} tiers total):")
    
    for tier_num in range(1, total_tiers + 1):
        tier_models = df[df['tier'] == tier_num]['model'].tolist()
        print(f"\n   Tier {tier_num}: {len(tier_models)} model(s)")
        for model in tier_models:
            score = df[df['model'] == model]['score'].values[0]
            print(f"      • {model} (score: {score:.4f})")
    
    print(f"\n" + "="*80 + "\n")
    
    # Create mapping from model name to tier
    return dict(zip(df['model'], df['tier']))


def create_plot(results: list[tuple], output_filename: str, open_models: set[str] | None = None) -> None:
    """Create a scatter plot of model performance vs. cost and save to PNG."""
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.lines import Line2D
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print("Plotting requires pandas and matplotlib. Install with:", file=sys.stderr)
        print("  pip install pandas matplotlib", file=sys.stderr)
        sys.exit(1)
    
    # Calculate average SD for models with sd=None
    valid_sds = [sd for _, _, sd, _, _ in results if sd is not None]
    avg_sd = sum(valid_sds) / len(valid_sds) if valid_sds else 0.0
    
    # Categorize models into tiers with debug output
    tier_mapping = categorize_tiers_debug(results, z_score=1.0)
    
    # Convert results to DataFrame
    df_data = []
    for model, avg, sd, n, cost in results:
        effective_sd = sd if sd is not None else avg_sd
        df_data.append({
            'Model Name': model,
            'Average Score': avg * 100,
            'Std Dev': effective_sd * 100,
            'Credit Cost (per 1k)': cost,
            'Tier': tier_mapping.get(model, 0)
        })
    
    df = pd.DataFrame(df_data)
    
    # Convert numeric columns (handling None in Cost)
    df['Credit Cost (per 1k)'] = pd.to_numeric(df['Credit Cost (per 1k)'], errors='coerce')
    
    # Filter out rows with NaN costs for the plot
    plot_df = df.dropna(subset=['Credit Cost (per 1k)'])
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Get colormap for tiers
    max_tier = plot_df['Tier'].max()
    colors = plt.cm.tab10(np.linspace(0, 1, max_tier))
    
    # Plot each tier separately with different colors
    for tier_num in sorted(plot_df['Tier'].unique()):
        tier_data = plot_df[plot_df['Tier'] == tier_num]
        color = colors[tier_num - 1]
        
        # Separate open and closed models within this tier
        open_data = tier_data[tier_data['Model Name'].isin(open_models)] if open_models else pd.DataFrame()
        closed_data = tier_data[~tier_data['Model Name'].isin(open_models)] if open_models else tier_data
        
        # Plot closed models (circles)
        if len(closed_data) > 0:
            plt.errorbar(closed_data['Credit Cost (per 1k)'], 
                         closed_data['Average Score'], 
                         yerr=closed_data['Std Dev'],
                         fmt='o',
                         color=color, 
                         ecolor=color,
                         alpha=0.7, 
                         markersize=10,
                         capsize=0,
                         markeredgecolor='black',
                         label=f'Tier {tier_num}')
        
        # Plot open models (squares)
        if len(open_data) > 0:
            plt.errorbar(open_data['Credit Cost (per 1k)'], 
                         open_data['Average Score'], 
                         yerr=open_data['Std Dev'],
                         fmt='s',
                         color=color, 
                         ecolor=color,
                         alpha=0.7, 
                         markersize=10,
                         capsize=0,
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
    plt.ylabel('Average Score $\\times 100$ (Lower is Better)', fontsize=19)
    plt.xlabel('Credit Cost (per 1k tokens) - Log Scale', fontsize=19)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add legend for tiers
    leg1 = plt.legend(loc='best', fontsize=12, title='Tiers', title_fontsize=13)
    plt.gca().add_artist(leg1)
    
    # Add legend for marker types
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=10, label='Proprietary models', markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
               markersize=10, label='Open models', markeredgecolor='black')
    ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=11)
    
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    
    # Use a log scale for X axis (Cost)
    plt.xscale('log')
    
    # Invert Y axis so the "Best" (Rank 1/Low Score) is at the top
    plt.gca().invert_yaxis()
    
    # Calculate y-axis limits
    max_score = max(avg for _, avg, _, _, _ in results) * 100
    upper_limit = get_y_upper_limit(max_score)
    plt.gca().set_ylim(top=0, bottom=upper_limit)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150)
    plt.close()
    
    print(f"Plot saved to: {output_filename}")


def parse_file(filename):
    """Parse ranking.txt and return (list_of_benchmark_dicts, cost_dict, open_dict)."""
    with open(filename, "r") as f:
        content = f.read()

    # Strip full-line comments
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
        description="Debug version: Compute unified model ranking with tiering details."
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
        print("Usage: python rank_models_debug.py [filename]", file=sys.stderr)
        print("  filename  Path to input file (default: ranking.txt)", file=sys.stderr)
        print("", file=sys.stderr)
        print("Run with 'ranking_sample.txt' for an example:", file=sys.stderr)
        print(f"  python rank_models_debug.py ranking_sample.txt", file=sys.stderr)
        sys.exit(1)

    benchmarks, cost_dict, open_dict = parse_file(filename)

    # Collect percentile scores per model
    model_scores: dict[str, list[float]] = {}

    for bench in benchmarks:
        total = bench.get("known_totals")
        if not total:
            continue
        for model, rank in bench.items():
            if model == "known_totals":
                continue
            if rank is None:
                continue
            pct = rank / total
            model_scores.setdefault(model, []).append(pct)

    # Compute average, std-dev, apply penalty
    results = []
    for model, scores in model_scores.items():
        n = len(scores)
        if n == 0:
            continue

        mean = sum(scores) / n

        # Sparse-data penalty
        penalty = 0.25 if n == 1 else (0.10 if n == 2 else 0.0)
        avg = mean + penalty

        # Population std-dev
        if n >= 2:
            var = sum((s - mean) ** 2 for s in scores) / n
            sd = math.sqrt(var)
        else:
            sd = None

        cost = cost_dict.get(model)
        results.append((model, avg, sd, n, cost))

    results.sort(key=lambda r: r[1])

    # Pretty-print ASCII table
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

    # Generate plot if requested (this will trigger debug output)
    if args.plot:
        base_name = os.path.splitext(filename)[0]
        plot_filename = f"{base_name}.png"
        open_models = set(open_dict.keys()) if open_dict else set()
        create_plot(results, plot_filename, open_models)
    else:
        # Still run tiering debug even without plot
        categorize_tiers_debug(results, z_score=1.0)


if __name__ == "__main__":
    main()
