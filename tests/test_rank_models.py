from itertools import pairwise
from pathlib import Path
import tomllib

import pytest

import rank_models
import ranking_math


def test_retired_benchmarks_are_excluded_from_active_data() -> None:
    """Keep archived benchmark data out of aggregate rankings."""
    data_dir = Path(rank_models.__file__).resolve().parent / "data"
    with (data_dir / "benchmarks.toml").open("rb") as f:
        active = tomllib.load(f)
    with (data_dir / "retired.toml").open("rb") as f:
        retired = tomllib.load(f)

    archived_names = {
        "GPQA_diamond",
        "MMMLU",
        "charxiv_notools",
        "swebench_pro_public",
        "swebench_verified",
    }
    assert archived_names <= retired.keys()
    assert not (set(active) & set(retired))


def test_fable5_is_absent_from_active_data() -> None:
    """Keep the retired model out of active rankings."""
    data_dir = Path(rank_models.__file__).resolve().parent / "data"
    with (data_dir / "benchmarks.toml").open("rb") as f:
        active = tomllib.load(f)
    with (data_dir / "models.toml").open("rb") as f:
        models = tomllib.load(f)

    assert all("fable5" not in benchmark["scores"] for benchmark in active.values())
    assert "fable5" not in models


def test_fable51_is_absent_from_active_data() -> None:
    """Keep Fable 5.1 out of active rankings while retaining Fable 5.2."""
    data_dir = Path(rank_models.__file__).resolve().parent / "data"
    with (data_dir / "benchmarks.toml").open("rb") as f:
        active = tomllib.load(f)
    with (data_dir / "models.toml").open("rb") as f:
        models = tomllib.load(f)

    assert all("fable51" not in benchmark["scores"] for benchmark in active.values())
    assert "fable51" not in models
    assert "fable52" in models
    assert any("fable52" in benchmark["scores"] for benchmark in active.values())


def test_muse_spark_12_is_absent_from_active_data() -> None:
    """Keep Muse Spark 12 out of active rankings while retaining Muse Spark 13."""
    data_dir = Path(rank_models.__file__).resolve().parent / "data"
    with (data_dir / "benchmarks.toml").open("rb") as f:
        active = tomllib.load(f)
    with (data_dir / "models.toml").open("rb") as f:
        models = tomllib.load(f)

    assert all(
        "muse-spark-12" not in benchmark["scores"] for benchmark in active.values()
    )
    assert "muse-spark-12" not in models
    assert "muse-spark-13" in models
    assert any("muse-spark-13" in benchmark["scores"] for benchmark in active.values())


def test_deepseek_flash4_is_absent_from_active_data() -> None:
    """Keep DeepSeek Flash 4 out of active rankings while retaining Flash 4.1."""
    data_dir = Path(rank_models.__file__).resolve().parent / "data"
    with (data_dir / "benchmarks.toml").open("rb") as f:
        active = tomllib.load(f)
    with (data_dir / "models.toml").open("rb") as f:
        models = tomllib.load(f)

    assert all(
        "deepseek-flash4" not in benchmark["scores"] for benchmark in active.values()
    )
    assert "deepseek-flash4" not in models
    assert "deepseek-flash41" in models
    assert any(
        "deepseek-flash41" in benchmark["scores"] for benchmark in active.values()
    )


def test_opus5_is_absent_from_active_data() -> None:
    """Keep Opus 5 out of active rankings while retaining Opus 5.5."""
    data_dir = Path(rank_models.__file__).resolve().parent / "data"
    with (data_dir / "benchmarks.toml").open("rb") as f:
        active = tomllib.load(f)
    with (data_dir / "models.toml").open("rb") as f:
        models = tomllib.load(f)

    assert all("opus5" not in benchmark["scores"] for benchmark in active.values())
    assert "opus5" not in models
    assert "opus55" in models
    assert any("opus55" in benchmark["scores"] for benchmark in active.values())


def _patch_data(
    monkeypatch: pytest.MonkeyPatch, benchmarks: list[dict[str, object]]
) -> None:
    costs = {
        "alpha": 10.0,
        "beta": 20.0,
        "gamma": 40.0,
        "solo": 5.0,
        "two": 15.0,
        "tie_a": 1.0,
        "tie_b": 2.0,
    }
    open_models = {"gamma": True}

    def fake_load_data(
        category: str,
    ) -> tuple[
        list[dict[str, object]],
        dict[str, float],
        dict[str, bool],
        dict[str, str],
        str,
        list[str],
    ]:
        return (
            benchmarks,
            costs,
            open_models,
            {},
            "Synthetic",
            [str(b.get("__name", f"bench-{i}")) for i, b in enumerate(benchmarks, 1)],
        )

    monkeypatch.setattr(rank_models, "load_data", fake_load_data)


def _result_by_model(
    results: list[tuple[object, ...]],
) -> dict[str, tuple[object, ...]]:
    return {str(row[0]): row for row in results}


def test_synthetic_three_benchmark_golden_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_data(
        monkeypatch,
        [
            {
                "__name": "score_one",
                "min_score": 0,
                "alpha": 100,
                "beta": 50,
                "gamma": 0,
            },
            {
                "__name": "rank_one",
                "known_totals": 10,
                "alpha": 2,
                "beta": 4,
                "gamma": 8,
            },
            {
                "__name": "score_two",
                "min_score": 0,
                "alpha": 80,
                "beta": 100,
                "gamma": 20,
            },
        ],
    )

    results, model_scores, *_ = rank_models._compute_raw("synthetic")
    by_model = _result_by_model(results)

    # Raw medians 0.2 / 0.4 / 0.8, so the prior (75th percentile) is 0.6. With n = 3 each
    # score is (3 x median + 0.6) / 4, and the raw IQR offsets are scaled by 3/4.
    assert [row[0] for row in results] == ["alpha", "beta", "gamma"]
    assert model_scores["alpha"] == pytest.approx([0.0, 0.2, 0.2])
    assert by_model["alpha"][1:5] == pytest.approx((0.3, 0.075, 0.0, 3))
    assert by_model["beta"][1:5] == pytest.approx((0.45, 0.15, 0.0375, 3))
    assert by_model["gamma"][1:5] == pytest.approx((0.75, 0.0, 0.075, 3))


def test_zero_range_min_score_benchmark_is_skipped_with_warning(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_data(
        monkeypatch,
        [
            {"__name": "flat_scores", "min_score": 10, "alpha": 10},
            {"__name": "rank_one", "known_totals": 10, "alpha": 2},
        ],
    )

    results, model_scores, *_ = rank_models._compute_raw("synthetic")
    captured = capsys.readouterr()

    assert (
        "Warning: benchmark 'flat_scores' skipped because score range is zero."
        in captured.err
    )
    assert model_scores["alpha"] == pytest.approx([0.2])
    assert results[0][0] == "alpha"


def test_all_none_benchmark_is_skipped_without_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_data(
        monkeypatch,
        [
            {"__name": "empty_scores", "min_score": 0, "alpha": None, "beta": None},
            {"__name": "rank_one", "known_totals": 10, "alpha": 1},
        ],
    )

    results, model_scores, *_ = rank_models._compute_raw("synthetic")

    assert model_scores == {"alpha": [0.1]}
    assert [row[0] for row in results] == ["alpha"]


def test_tied_scores_have_deterministic_order_and_tier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_data(
        monkeypatch,
        [
            {"__name": "tie_scores", "min_score": 0, "tie_a": 100, "tie_b": 100},
            {"__name": "tie_ranks", "known_totals": 10, "tie_a": 1, "tie_b": 1},
            {"__name": "tie_scores_two", "min_score": 0, "tie_a": 50, "tie_b": 50},
        ],
    )

    raw = rank_models._compute_raw("synthetic")
    tiers = rank_models.categorize_tiers(raw.results, raw.replicates)

    assert [row[0] for row in raw.results] == ["tie_a", "tie_b"]
    assert tiers == {"tie_a": 1, "tie_b": 1}


def test_sparse_models_are_shrunk_and_have_no_iqr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_data(
        monkeypatch,
        [
            {"__name": "rank_one", "known_totals": 10, "solo": 1, "two": 2},
            {"__name": "rank_two", "known_totals": 10, "solo": None, "two": 4},
        ],
    )

    results, *_ = rank_models._compute_raw("synthetic")
    by_model = _result_by_model(results)

    # Raw medians 0.1 (n=1) and 0.3 (n=2); prior = 75th percentile = 0.25.
    assert by_model["solo"][1] == pytest.approx((0.1 + 0.25) / 2)
    assert by_model["solo"][2] is None
    assert by_model["solo"][3] is None
    assert by_model["solo"][4] == 1
    assert by_model["two"][1] == pytest.approx((2 * 0.3 + 0.25) / 3)
    assert by_model["two"][2] is None
    assert by_model["two"][3] is None
    assert by_model["two"][4] == 2


def test_mixed_min_score_and_known_totals_aggregate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_data(
        monkeypatch,
        [
            {"__name": "score_schema", "min_score": 0, "alpha": 100, "beta": 50},
            {"__name": "rank_schema", "known_totals": 20, "alpha": 4, "beta": 10},
        ],
    )

    results, model_scores, *_ = rank_models._compute_raw("synthetic")
    by_model = _result_by_model(results)

    assert model_scores["alpha"] == pytest.approx([0.0, 0.2])
    assert model_scores["beta"] == pytest.approx([0.5, 0.5])
    # Raw medians 0.1 and 0.5; prior = 75th percentile = 0.4; n = 2 for both.
    assert by_model["alpha"][1] == pytest.approx((2 * 0.1 + 0.4) / 3)
    assert by_model["beta"][1] == pytest.approx((2 * 0.5 + 0.4) / 3)
    assert [row[0] for row in results] == ["alpha", "beta"]


def test_malformed_benchmarks_toml_errors_clearly(tmp_path) -> None:  # type: ignore[no-untyped-def]
    malformed = tmp_path / "bad.toml"
    malformed.write_text("[broken\nvalue = 1\n")

    with pytest.raises(SystemExit) as exc_info:
        rank_models._load_benchmarks_toml(str(malformed))

    assert "Failed to parse TOML file" in str(exc_info.value)
    assert str(malformed) in str(exc_info.value)


def test_compute_rankings_export_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_data(
        monkeypatch,
        [
            {"__name": "score_one", "min_score": 0, "alpha": 100, "beta": 50},
            {"__name": "rank_one", "known_totals": 10, "alpha": 2, "beta": 4},
            {"__name": "score_two", "min_score": 0, "alpha": 80, "beta": 100},
        ],
    )

    exported = rank_models.compute_rankings("synthetic")

    assert {
        "category",
        "title",
        "generated_at",
        "n_benchmarks",
        "benchmark_names",
        "palette",
        "models",
    } <= exported.keys()
    assert exported["category"] == "synthetic"
    assert exported["title"] == "Synthetic"
    assert exported["n_benchmarks"] == 3
    assert exported["benchmark_names"] == ["score_one", "rank_one", "score_two"]

    assert exported["uncertainty"]["method"] == "benchmark bootstrap"
    assert exported["shrinkage"] == {
        "k": ranking_math.SHRINK_K,
        "prior_quantile": ranking_math.SHRINK_QUANTILE,
        "prior": pytest.approx(
            ranking_math.shrinkage_prior(
                {m["name"]: m["raw_scores"] for m in exported["models"]}
            )
        ),
    }
    for row in exported["models"]:
        assert row["median_pct"] == pytest.approx(sorted(row["raw_scores"])[1])

    model = exported["models"][0]
    assert model["rank_lo"] <= model["rank"] <= model["rank_hi"]
    assert {
        "rank",
        "rank_lo",
        "rank_hi",
        "name",
        "avg_pct",
        "lower_err",
        "upper_err",
        "n_bench",
        "cost",
        "rel_cost",
        "tier",
        "is_open",
        "company",
        "raw_scores",
    } <= model.keys()


def _patch_toml(
    monkeypatch: pytest.MonkeyPatch, benchmarks: dict[str, dict[str, object]]
) -> None:
    monkeypatch.setattr(
        rank_models, "_load_benchmarks_toml", lambda _: list(benchmarks.items())
    )
    monkeypatch.setattr(rank_models, "_load_models_toml", lambda _: {})


def test_score_below_floor_exits_naming_benchmark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_toml(
        monkeypatch,
        {
            "Physics": {
                "categories": ["stem"],
                "min_score": 8,
                "scores": {"alpha": 27.4, "beta": 4.2},
            }
        },
    )

    with pytest.raises(SystemExit) as exc_info:
        rank_models.load_data("stem")

    assert "'Physics'" in str(exc_info.value)
    assert "beta=4.2" in str(exc_info.value)
    assert "alpha" not in str(exc_info.value)


def test_rank_outside_known_totals_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_toml(
        monkeypatch,
        {
            "Leaderboard": {
                "categories": ["general"],
                "known_totals": 10,
                "scores": {"alpha": 1, "beta": 11},
            }
        },
    )

    with pytest.raises(SystemExit) as exc_info:
        rank_models.load_data("general")

    assert "'Leaderboard'" in str(exc_info.value)
    assert "beta=11" in str(exc_info.value)


def _spread_benchmarks() -> list[dict[str, object]]:
    """Ten benchmarks: six fillers, a model on all ten, and a noisy model on three."""
    benchmarks: list[dict[str, object]] = []
    for i in range(10):
        bench: dict[str, object] = {"__name": f"b{i}", "min_score": 0}
        for j, base in enumerate([90, 80, 70, 60, 50, 40]):
            bench[f"filler{j}"] = base + (i * 7 + j * 3) % 9 - 4
        bench["thick"] = 65 + (i % 3) - 1
        if i < 3:
            bench["thin"] = [95, 30, 66][i]
        benchmarks.append(bench)
    return benchmarks


def test_bootstrap_is_deterministic_and_thin_models_get_wider_rank_intervals() -> None:
    benchmarks = _spread_benchmarks()
    results, _ = ranking_math.calculate_rankings(benchmarks)

    first = ranking_math.bootstrap_scores(benchmarks)
    second = ranking_math.bootstrap_scores(benchmarks)
    assert first == second

    intervals = ranking_math.rank_intervals(first, results)
    thin_lo, thin_hi = intervals["thin"]
    thick_lo, thick_hi = intervals["thick"]
    assert thin_hi - thin_lo > thick_hi - thick_lo
    for position, row in enumerate(results, 1):
        lo, hi = intervals[row[0]]
        assert lo <= position <= hi


def test_shrinkage_fades_smoothly_with_coverage() -> None:
    """Same raw median on 1, 2, 3 and 6 benchmarks: each extra benchmark helps, by less."""
    counts = (1, 2, 3, 6)
    benchmarks: list[dict[str, object]] = []
    for i in range(6):
        bench: dict[str, object] = {
            "__name": f"b{i}",
            "min_score": 0,
            "leader": 100,
            "weak_a": 40,
            "weak_b": 20,
        }
        bench.update({f"m{n}": 80 for n in counts if i < n})
        benchmarks.append(bench)

    results, model_scores = ranking_math.calculate_rankings(benchmarks)
    score = {row[0]: row[1] for row in results}

    # Raw medians sorted: 0 (leader), 0.2 x 4 (m<n>), 0.6, 0.8 -> 75th percentile = 0.4.
    prior = ranking_math.shrinkage_prior(model_scores)
    assert prior is not None
    assert prior == pytest.approx(0.4)
    for n in counts:
        assert score[f"m{n}"] == pytest.approx((n * 0.2 + prior) / (n + 1))

    per_benchmark_gain = [
        (score[f"m{a}"] - score[f"m{b}"]) / (b - a) for a, b in pairwise(counts)
    ]
    assert all(gain > 0 for gain in per_benchmark_gain)
    assert per_benchmark_gain == sorted(per_benchmark_gain, reverse=True)
    # The model at the top of its benchmarks is pulled toward the prior too, but less.
    assert 0 < score["leader"] < score["m6"]


@pytest.mark.parametrize("category", sorted(rank_models.CATEGORIES))
def test_shrinkage_keeps_raw_median_order_among_equal_coverage(category: str) -> None:
    exported = rank_models.compute_rankings(category)
    by_n: dict[int, list[float]] = {}
    for model in exported["models"]:  # already sorted by shrunk score
        by_n.setdefault(model["n_bench"], []).append(model["median_pct"])
    for medians in by_n.values():
        assert medians == sorted(medians)


def test_models_that_never_share_a_resample_are_indistinguishable() -> None:
    replicates = [{"a": 0.1}, {"b": 0.5}]
    assert ranking_math.prob_at_least_as_good(replicates, "b", "a") == 1.0


@pytest.mark.parametrize("category", sorted(rank_models.CATEGORIES))
def test_real_data_tiers_are_contiguous_in_rank_order(category: str) -> None:
    exported = rank_models.compute_rankings(category)
    tiers = [model["tier"] for model in exported["models"]]

    assert tiers[0] == 1
    assert all(b - a in (0, 1) for a, b in pairwise(tiers))
    for model in exported["models"]:
        assert model["rank_lo"] <= model["rank"] <= model["rank_hi"]
