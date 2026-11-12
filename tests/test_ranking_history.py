"""Check historical reconstruction, provenance, and animation semantics."""

from pathlib import Path

import pytest

import animate_rankings as animation
import rank_models
import ranking_history as history
from ranking_math import calculate_rankings


def snapshot(day, scores, version="original", benchmark="SWE_Marathon"):
    return history.Snapshot(
        day,
        day.replace("-", ""),
        "data/benchmarks.toml",
        [{"__name": benchmark, "min_score": 0, **scores}],
        {benchmark: version},
    )


@pytest.mark.parametrize(
    "path,text",
    [
        (
            "data/general.txt",
            "HLE = {'min_score': 0, 'model-v1': 42, 'absent': None}\n"
            "cost = {'model-v1': 1}\n{'metadata': 3}",
        ),
        (
            "data/benchmarks.txt",
            "HLE = {'categories': ['general'], 'min_score': 0, "
            "'scores': {'model-v1': 42, 'absent': None}}",
        ),
        (
            "data/benchmarks.toml",
            '[HLE]\ncategories=["general"]\nmin_score=0\n[HLE.scores]\nmodel-v1=42',
        ),
    ],
)
def test_all_historical_formats(path, text):
    benches = history.parse_benchmarks(text, path, "general")
    assert len(benches) == 1
    assert benches[0]["model-v1"] == 42
    # A lone model is its own shrinkage prior, so its score is its raw gap of 0.
    assert calculate_rankings(benches)[0][0][1] == 0.0


def test_chained_assignment_counts_benchmark_once():
    benches = history.parse_benchmarks(
        "terminal_bench = result = {'known_totals': 10, 'a': 1}",
        "data/coding.txt",
        "coding",
    )
    assert len(benches) == 1
    assert benches[0]["__name"] == "terminal_bench"


@pytest.mark.parametrize(
    "payload",
    [
        "import os",
        "a = __import__('os').getcwd()",
        "a = dict(a=1)",
        "a = {'min_score': 0, 'model': float('nan')}",
    ],
)
def test_legacy_code_never_executes(payload):
    with pytest.raises((ValueError, SyntaxError)):
        history.parse_literals(payload)


def test_malformed_toml_only_affects_its_category():
    text = (
        '[HLE]\ncategories=["general"]\nmin_score=0\n[HLE.scores]\na=42\n'
        '[CritPt]\ncategories=["stem"]\nmin_score=0\n[CritPt.scores]\na=None'
    )
    assert (
        history.parse_benchmarks(text, "data/benchmarks.toml", "general")[0]["a"] == 42
    )
    with pytest.raises(ValueError):
        history.parse_benchmarks(text, "data/benchmarks.toml", "stem")


def test_same_day_last_snapshot_and_unchanged_collapse():
    first = snapshot("2026-06-01", {"a": 40})
    last = snapshot("2026-06-01", {"b": 60})
    final = snapshot("2026-06-02", {"b": 60})
    daily = history.daily_snapshots([first, last, final])
    assert daily == [last, final]
    states = history.progress_states(daily, "SWE_Marathon")
    assert all(m["name"] != "a" for s in states for m in s["models"])
    assert states[0]["sources"][-1]["date"] == "2026-06-02"
    assert states[-1]["date"] == "2026-06-02"


def test_read_history_logs_broken_final_day_without_substitution(monkeypatch, tmp_path):
    blobs = {
        "a": "HLE={'min_score':0, 'one':40}",
        "b": "HLE={broken",
        "c": "HLE={'min_score':0, 'future':90}",
    }

    def fake_git(repo, *args):
        if args[0] == "log":
            return "a 2026-03-01\nb 2026-03-01\nc 2026-03-02"
        if args[0] == "ls-tree":
            return f"100644 blob {args[-1]}\tdata/general.txt"
        return blobs[args[-1]]

    monkeypatch.setattr(history, "git", fake_git)
    snapshots, skipped = history.read_history(tmp_path)
    assert [s.commit for s in snapshots["general"]] == ["c"]
    assert skipped[0]["commit"] == "b"
    assert skipped[0]["category"] == "general"
    assert skipped[0]["reason"]


def test_retention_corrections_missing_observations_and_gains():
    observations = [
        snapshot("2026-06-01", {"old": 60, "corrected": 50}),
        snapshot("2026-06-02", {"corrected": 40, "new": 70}),
        snapshot("2026-06-03", {"new": 55, "corrected": None}),
    ]
    states = history.progress_states(observations, "SWE_Marathon")
    assert [s["best"] for s in states] == [60, 70, 60]
    assert [s["gain"] for s in states] == [0, 10, 0]
    assert "new" not in {m["name"] for m in states[0]["models"]}
    final = {m["name"]: m for m in states[-1]["models"]}
    assert final["corrected"]["value"] == 40
    assert final["corrected"]["observed_date"] == "2026-06-02"
    assert final["old"]["retained"]
    assert final["old"]["observed_commit"] == "20260601"
    assert not final["new"]["retained"]


def test_version_boundary_resets_models_and_baseline():
    observations = [
        snapshot("2026-06-01", {"old": 60, "same": 40}),
        snapshot("2026-08-01", {"same": 20, "new": 35}, "v1.1"),
        snapshot("2026-08-02", {"new": 30}, "v1.1"),
    ]
    states = history.progress_states(observations, "SWE_Marathon")
    assert "old" not in {m["name"] for m in states[1]["models"]}
    assert states[1]["baseline"] == 35
    assert states[-1]["gain"] == -5
    assert (
        history.benchmark_versions(
            "# https://www.swe-marathon.org, v1.1\n[SWE_Marathon]\n"
        )["SWE_Marathon"]
        == "v1.1"
    )
    assert (
        history.benchmark_versions("# v1.1\n[DeepSWE]\nx=1\n\n# URL\n[SWE_Marathon]\n")[
            "SWE_Marathon"
        ]
        == "original"
    )


def test_rank_only_hle_not_mistaken_for_raw_scores():
    early = snapshot("2026-03-01", {"a": 1}, benchmark="HLE")
    early.benchmarks = [{"__name": "HLE", "known_totals": 100, "a": 1}]
    later = snapshot("2026-03-14", {"a": 40}, benchmark="HLE")
    states = history.progress_states([early, later], "HLE")
    assert len(states) == 1
    assert states[0]["date"] == "2026-03-14"
    assert states[0]["best"] == 40


def test_aggregate_removes_models_and_retired_benchmarks():
    old = snapshot("2026-06-01", {"old": 60})
    new = snapshot("2026-06-02", {"new": 70}, benchmark="HLE")
    states = history.ranking_states([old, new])
    assert states[-1]["removed"] == ["SWE_Marathon"]
    assert states[-1]["added"] == ["HLE"]
    assert [m["name"] for m in states[-1]["models"]] == ["new"]
    assert states[-1]["models"][0]["value"] == 100
    assert states[-1]["models"][0]["sparse"]


def test_progress_retains_retired_benchmark():
    old = snapshot("2026-06-01", {"old": 60})
    new = snapshot("2026-06-02", {"new": 70}, benchmark="HLE")
    states = history.progress_states([old, new], "SWE_Marathon")
    assert states[-1]["models"][0]["retained"]
    assert states[-1]["models"][0]["observed_date"] == "2026-06-01"


def test_latest_reconstruction_matches_existing_cli_all_categories():
    snapshots, _ = history.read_history(Path(__file__).resolve().parents[1])
    for category in history.CATEGORIES:
        old, old_scores, *_ = rank_models._compute_raw(category)
        new, new_scores = calculate_rankings(snapshots[category][-1].benchmarks)
        assert [r[:5] for r in new] == [r[:5] for r in old]
        assert new_scores == old_scores


def test_entrants_fade_at_first_score_and_departures_fade_out():
    states = history.progress_states(
        [snapshot("2026-06-01", {"old": 40}), snapshot("2026-06-02", {"new": 70})],
        "SWE_Marathon",
    )
    rows = {r["name"]: r for r in animation.interpolate_models(*states, 0.25, 1)}
    assert rows["new"]["value"] == 70
    assert rows["new"]["alpha"] == 0.25
    assert rows["old"]["alpha"] == 0.75


def test_chapters_do_not_interpolate_measurements():
    states = history.progress_states(
        [
            snapshot("2026-06-01", {"same": 60}),
            snapshot("2026-08-01", {"same": 20}, "v1.1"),
        ],
        "SWE_Marathon",
    )
    assert animation.interpolate_models(*states, 0.25, 10)[0]["value"] == 60
    assert animation.interpolate_models(*states, 0.75, 10)[0]["value"] == 20


def test_animation_holds_and_final_frame():
    assert animation.frame_position(0, 1800, 20, 30) == (0, 0, 1)
    assert animation.frame_position(1799, 1800, 20, 30) == (19, 19, 1)
    assert animation.frame_position(900, 1800, 1, 30) == (0, 0, 1)


def test_empty_first_observation_in_new_chapter_resets_baseline():
    states = history.progress_states(
        [
            snapshot("2026-06-01", {"old": 60}),
            snapshot("2026-08-01", {"new": None}, "v1.1"),
            snapshot("2026-08-02", {"new": 30}, "v1.1"),
        ],
        "SWE_Marathon",
    )
    assert states[-1]["baseline"] == 30
    assert states[-1]["gain"] == 0


def test_date_window_filters_before_collapse_preserving_prior_observations():
    states = history.progress_states(
        [
            snapshot("2026-06-01", {"old": 60}),
            snapshot("2026-06-02", {"new": 40}),
            snapshot("2026-06-03", {"new": 40}),
        ],
        "SWE_Marathon",
        collapse=False,
    )
    window = history.collapse_states([s for s in states if s["date"] == "2026-06-03"])
    assert len(window) == 1
    assert window[0]["models"][0]["observed_date"] == "2026-06-01"
