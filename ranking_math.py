"""Shared, standard-library-only normalization, median aggregation, and bootstrap tiering."""

from bisect import bisect_left
import math
import random
import statistics
import sys
from typing import Any

BENCHMARK_META_KEYS = {"known_totals", "min_score", "__name"}
# (model, score, lower_err, upper_err, n_benchmarks, cost); score is the shrunk median gap.
RankingResult = tuple[str, float, float | None, float | None, int, float | None]

# Coverage shrinkage: a model's median gap is blended with a phantom result worth SHRINK_K
# benchmarks, set at the SHRINK_QUANTILE of all models' medians (0.75 = below average).
# Own data gets weight n / (n + SHRINK_K), so thin evidence counts for less, smoothly.
SHRINK_K = 1.0
SHRINK_QUANTILE = 0.75

# One bootstrap replicate: model -> aggregate score. Models with no resampled data are absent.
Replicate = dict[str, float]
# One tiering comparison: (leader, candidate, P(candidate at least as good), joined tier).
TierStep = tuple[str, str, float, bool]

N_BOOT = 2000
BOOT_SEED = 42
# Central share of bootstrap ranks reported as the rank interval (~ +/-1 sigma).
RANK_INTERVAL = 0.68
# A candidate joins the leader's tier while it matches or beats the leader in at least
# this share of resamples (the one-sided tail of a ~1 sigma interval).
TIER_ALPHA = 0.16


def _quantile(sorted_values: list[float], q: float) -> float:
    """Return the q-quantile of an ascending list by linear interpolation between neighbours."""
    idx = (len(sorted_values) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (idx - lo) * (sorted_values[hi] - sorted_values[lo])


def shrinkage_prior(model_scores: dict[str, list[float]]) -> float | None:
    """Return the phantom result models are shrunk toward: a quantile of all models' medians."""
    medians = sorted(statistics.median(s) for s in model_scores.values() if s)
    return _quantile(medians, SHRINK_QUANTILE) if medians else None


def shrink(value: float, n: int, prior: float) -> float:
    """Blend a model's own value (from n benchmarks) with the prior worth SHRINK_K benchmarks."""
    return (n * value + SHRINK_K * prior) / (n + SHRINK_K)


def calculate_rankings(
    benchmarks: list[dict[str, Any]],
    cost_dict: dict[str, float] | None = None,
    warn: bool = True,
) -> tuple[list[RankingResult], dict[str, list[float]]]:
    """Calculate normalized gaps, shrunk medians, and asymmetric IQR bounds.

    Returns ranked tuples (best first) and each model's raw per-benchmark gaps.
    """
    cost_dict = cost_dict or {}
    model_scores: dict[str, list[float]] = {}

    for bench in benchmarks:
        benchmark_name = str(bench.get("__name", "unnamed benchmark"))
        min_score = bench.get("min_score")
        if min_score is not None:
            observed = [
                v
                for k, v in bench.items()
                if k not in BENCHMARK_META_KEYS and v is not None
            ]
            if not observed:
                continue
            max_score = max(observed)
            score_range = max_score - min_score
            if score_range == 0:
                if warn:
                    print(
                        f"Warning: benchmark '{benchmark_name}' skipped because score range is zero.",
                        file=sys.stderr,
                    )
                continue
            for model, score in bench.items():
                if model in BENCHMARK_META_KEYS:
                    continue
                if score is None:
                    continue
                pct = (max_score - score) / score_range
                model_scores.setdefault(model, []).append(pct)
        else:
            total = bench.get("known_totals")
            if not total:
                continue
            for model, rank in bench.items():
                if model in BENCHMARK_META_KEYS:
                    continue
                if rank is None:
                    continue
                pct = rank / total
                model_scores.setdefault(model, []).append(pct)

    prior = shrinkage_prior(model_scores)
    results: list[RankingResult] = []
    for model, scores in model_scores.items():
        n = len(scores)
        if n == 0 or prior is None:
            continue

        sorted_scores = sorted(scores)
        median = statistics.median(sorted_scores)
        avg = min(shrink(median, n, prior), 1.0)

        if n >= 3:
            # Blending Q1 and Q3 the same way scales the bars by n / (n + k), so they
            # still bracket the shrunk score.
            weight = n / (n + SHRINK_K)
            lower_err = weight * (median - _quantile(sorted_scores, 0.25))
            upper_err = weight * (_quantile(sorted_scores, 0.75) - median)
        else:
            lower_err = None
            upper_err = None

        cost = cost_dict.get(model)
        results.append((model, avg, lower_err, upper_err, n, cost))

    results.sort(key=lambda r: r[1])
    return results, model_scores


def bootstrap_scores(
    benchmarks: list[dict[str, Any]], n_boot: int = N_BOOT, seed: int = BOOT_SEED
) -> list[Replicate]:
    """Resample benchmarks with replacement and recompute every model's aggregate score.

    Each replicate draws as many benchmarks as the category has, with replacement, and
    reruns calculate_rankings on that draw. All models in a replicate share the same draw,
    so comparisons between models are paired. A model none of whose benchmarks were drawn
    is absent from that replicate. A fixed seed keeps the output reproducible.
    """
    if not benchmarks:
        return []
    rng = random.Random(seed)
    k = len(benchmarks)
    replicates: list[Replicate] = []
    for _ in range(n_boot):
        draw = [benchmarks[rng.randrange(k)] for _ in range(k)]
        results, _ = calculate_rankings(draw, warn=False)
        replicates.append({r[0]: r[1] for r in results})
    return replicates


def rank_intervals(
    replicates: list[Replicate],
    results: list[RankingResult],
    level: float = RANK_INTERVAL,
) -> dict[str, tuple[int, int]]:
    """Return each model's central bootstrap rank interval as (best rank, worst rank).

    Within a replicate, a model's rank is 1 + the number of present models scoring
    strictly better. Replicates where the model is absent are skipped. The interval is
    widened if needed so it always contains the model's listed rank in `results`.
    """
    ranks: dict[str, list[int]] = {r[0]: [] for r in results}
    for rep in replicates:
        ordered = sorted(rep.values())
        for model, score in rep.items():
            if model in ranks:
                ranks[model].append(bisect_left(ordered, score) + 1)

    lo_q = (1 - level) / 2
    hi_q = (1 + level) / 2
    intervals: dict[str, tuple[int, int]] = {}
    for position, row in enumerate(results, 1):
        model = row[0]
        rs = sorted(ranks[model])
        if not rs:
            intervals[model] = (position, position)
            continue
        lo = rs[math.floor(lo_q * (len(rs) - 1))]
        hi = rs[math.ceil(hi_q * (len(rs) - 1))]
        intervals[model] = (min(lo, position), max(hi, position))
    return intervals


def prob_at_least_as_good(
    replicates: list[Replicate], model: str, reference: str
) -> float:
    """Return the share of replicates (with both models present) where `model` <= `reference`.

    Lower scores are better. Returns 1.0 when the two models never appear together,
    because there is then no evidence that they differ.
    """
    shared = 0
    wins = 0
    for rep in replicates:
        if model in rep and reference in rep:
            shared += 1
            if rep[model] <= rep[reference]:
                wins += 1
    return wins / shared if shared else 1.0


def assign_tiers(
    results: list[RankingResult],
    replicates: list[Replicate],
    alpha: float = TIER_ALPHA,
) -> tuple[dict[str, int], list[TierStep]]:
    """Group models into tiers that are contiguous in rank order.

    The best remaining model leads a new tier. Walking down the ranking, each next model
    joins while it matches or beats the leader in at least `alpha` of the resamples; the
    first model that does not becomes the next leader. Returns the tier mapping and the
    comparisons made, for debug output.
    """
    order = [r[0] for r in results]
    tiers: dict[str, int] = {}
    steps: list[TierStep] = []
    tier = 0
    i = 0
    while i < len(order):
        tier += 1
        leader = order[i]
        tiers[leader] = tier
        i += 1
        while i < len(order):
            p = prob_at_least_as_good(replicates, order[i], leader)
            joined = p >= alpha
            steps.append((leader, order[i], p, joined))
            if not joined:
                break
            tiers[order[i]] = tier
            i += 1
    return tiers, steps
