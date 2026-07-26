from pathlib import Path
import tomllib

import pytest

import rank_models


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
    assert "charxiv" in active
    assert "arena_webdev" in active
    assert "arena_coding" not in active
    assert active["ARC_AGI_2"]["categories"] == ["general"]
    assert active["osworld"]["categories"] == ["agentic"]
    assert active["GraphWalks_BFS_1M"]["categories"] == ["general"]
    assert active["FrontierCode"]["min_score"] == pytest.approx(24.31)
    assert active["FrontierCode"]["scores"] == {
        "fable5": 53.48,
        "gpt55": 42.96,
        "opus48": 46.5,
        "sonnet5": 42.73,
    }


def _patch_data(monkeypatch: pytest.MonkeyPatch, benchmarks: list[dict[str, object]]) -> None:
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
    ) -> tuple[list[dict[str, object]], dict[str, float], dict[str, bool], dict[str, str], str, list[str]]:
        return benchmarks, costs, open_models, {}, "Synthetic", [
            str(b.get("__name", f"bench-{i}")) for i, b in enumerate(benchmarks, 1)
        ]

    monkeypatch.setattr(rank_models, "load_data", fake_load_data)


def _result_by_model(results: list[tuple[object, ...]]) -> dict[str, tuple[object, ...]]:
    return {str(row[0]): row for row in results}


def test_synthetic_three_benchmark_golden_scores(monkeypatch: pytest.MonkeyPatch) -> None:
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

    assert [row[0] for row in results] == ["alpha", "beta", "gamma"]
    assert model_scores["alpha"] == pytest.approx([0.0, 0.2, 0.2])
    assert by_model["alpha"][1:5] == pytest.approx((0.2, 0.1, 0.0, 3))
    assert by_model["beta"][1:5] == pytest.approx((0.4, 0.2, 0.05, 3))
    assert by_model["gamma"][1:5] == pytest.approx((0.8, 0.0, 0.1, 3))


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

    assert "Warning: benchmark 'flat_scores' skipped because score range is zero." in captured.err
    assert model_scores["alpha"] == pytest.approx([0.2])
    assert results[0][0] == "alpha"


def test_all_none_benchmark_is_skipped_without_crash(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_tied_scores_have_deterministic_order_and_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_data(
        monkeypatch,
        [
            {"__name": "tie_scores", "min_score": 0, "tie_a": 100, "tie_b": 100},
            {"__name": "tie_ranks", "known_totals": 10, "tie_a": 1, "tie_b": 1},
            {"__name": "tie_scores_two", "min_score": 0, "tie_a": 50, "tie_b": 50},
        ],
    )

    results, *_ = rank_models._compute_raw("synthetic")
    tiers = rank_models.categorize_tiers(results)

    assert [row[0] for row in results] == ["tie_a", "tie_b"]
    assert tiers == {"tie_a": 1, "tie_b": 1}


def test_sparse_data_penalties_and_missing_iqr(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_data(
        monkeypatch,
        [
            {"__name": "rank_one", "known_totals": 10, "solo": 1, "two": 2},
            {"__name": "rank_two", "known_totals": 10, "solo": None, "two": 4},
        ],
    )

    results, *_ = rank_models._compute_raw("synthetic")
    by_model = _result_by_model(results)

    assert by_model["solo"][1] == pytest.approx(0.35)
    assert by_model["solo"][2] is None
    assert by_model["solo"][3] is None
    assert by_model["solo"][4] == 1
    assert by_model["two"][1] == pytest.approx(0.4)
    assert by_model["two"][2] is None
    assert by_model["two"][3] is None
    assert by_model["two"][4] == 2


def test_mixed_min_score_and_known_totals_aggregate(monkeypatch: pytest.MonkeyPatch) -> None:
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
    assert by_model["alpha"][1] == pytest.approx(0.2)
    assert by_model["beta"][1] == pytest.approx(0.6)
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

    model = exported["models"][0]
    assert {
        "rank",
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
