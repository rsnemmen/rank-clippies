"""Shared, standard-library-only normalization and median aggregation."""

import sys
from typing import Any

BENCHMARK_META_KEYS = {"known_totals", "min_score", "__name"}
RankingResult = tuple[str, float, float | None, float | None, int, float | None]


def calculate_rankings(
    benchmarks: list[dict[str, Any]], cost_dict: dict[str, float] | None = None
) -> tuple[list[RankingResult], dict[str, list[float]]]:
    """Calculate normalized medians, sparse penalties, and asymmetric IQR bounds."""
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

    results: list[RankingResult] = []
    for model, scores in model_scores.items():
        n = len(scores)
        if n == 0:
            continue

        sorted_scores = sorted(scores)
        mid = n // 2
        median = (
            sorted_scores[mid]
            if n % 2 == 1
            else (sorted_scores[mid - 1] + sorted_scores[mid]) / 2
        )

        penalty = 0.25 if n == 1 else (0.10 if n == 2 else 0.0)
        avg = min(median + penalty, 1.0)

        if n >= 3:
            q1_idx = (n - 1) * 0.25
            q3_idx = (n - 1) * 0.75
            q1 = sorted_scores[int(q1_idx)] + (q1_idx % 1) * (
                sorted_scores[min(int(q1_idx) + 1, n - 1)] - sorted_scores[int(q1_idx)]
            )
            q3 = sorted_scores[int(q3_idx)] + (q3_idx % 1) * (
                sorted_scores[min(int(q3_idx) + 1, n - 1)] - sorted_scores[int(q3_idx)]
            )
            lower_err = median - q1
            upper_err = q3 - median
        else:
            lower_err = None
            upper_err = None

        cost = cost_dict.get(model)
        results.append((model, avg, lower_err, upper_err, n, cost))

    results.sort(key=lambda r: r[1])
    return results, model_scores
