"""Reconstruct committed observations without checking out or executing old code."""

import ast
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import subprocess
import tomllib
from typing import Any

from ranking_math import BENCHMARK_META_KEYS, calculate_rankings

CATEGORIES = ("general", "coding", "agentic", "stem")
DEFAULT_BENCHMARKS = {
    "general": "HLE",
    "coding": "SWE_Marathon",
    "agentic": "ARC_AGI_2",
    "stem": "CritPt",
}
BENCHMARK_LABELS = {
    "HLE": "HLE · without tools",
    "SWE_Marathon": "SWE-Marathon",
    "ARC_AGI_2": "ARC-AGI-2",
    "CritPt": "CritPt",
}


@dataclass
class Snapshot:
    """Store one category's committed inputs and their provenance."""

    date: str
    commit: str
    path: str
    benchmarks: list[dict[str, Any]]
    versions: dict[str, str]


def git(repo: Path, *args: str) -> str:
    """Read Git output without modifying the working tree."""
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


def parse_literals(text: str) -> dict[str, Any]:
    """Parse legacy assignments and bare metadata dictionaries using literal_eval."""
    result: dict[str, Any] = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign):
            # Chained legacy labels (terminal_bench = result = {...}) are one benchmark.
            if any(not isinstance(t, ast.Name) for t in node.targets):
                raise ValueError("legacy assignment target is not a name")
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                raise ValueError("legacy assignment target is not a name")
            result[target.id] = ast.literal_eval(node.value)
        elif isinstance(node, ast.Expr):
            # Earliest files use an unassigned cost dictionary; never execute it.
            ast.literal_eval(node.value)
        else:
            raise ValueError("legacy file contains a non-literal statement")
    return result


def numeric(value: Any, label: str) -> float:
    """Validate a finite numeric measurement."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: expected a number")
    if not math.isfinite(value):
        raise ValueError(f"{label}: non-finite value")
    return float(value)


def parse_benchmarks(text: str, path: str, category: str) -> list[dict[str, Any]]:
    """Read any historical layout, preserving input order and exact model keys."""
    if path.endswith(".toml"):
        raw: dict[str, Any] = {}
        # Isolate top-level benchmark blocks so invalid STEM data does not erase coding.
        starts = list(re.finditer(r"^\[([^\]\n.]+)\]\s*$", text, re.MULTILINE))
        for index, match in enumerate(starts):
            end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
            block = text[match.start() : end]
            try:
                parsed = tomllib.loads(block)
                if raw.keys() & parsed.keys():
                    raise ValueError("duplicate benchmark table")
                raw.update(parsed)
            except tomllib.TOMLDecodeError:
                cats = re.search(r"^categories\s*=.*$", block, re.MULTILINE)
                if cats and category not in tomllib.loads(cats[0]).get(
                    "categories", []
                ):
                    continue
                raise
        if not starts:
            raise ValueError("no benchmark tables")
    else:
        raw = parse_literals(text)
    centralized = Path(path).name in ("benchmarks.txt", "benchmarks.toml")
    benchmarks: list[dict[str, Any]] = []
    for name, entry in raw.items():
        if not centralized and name in ("cost", "open", "title"):
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"{name}: expected a benchmark dictionary")
        if centralized:
            cats = entry.get("categories")
            if (
                not isinstance(cats, list)
                or not cats
                or any(c not in CATEGORIES for c in cats)
            ):
                raise ValueError(f"{name}: invalid categories")
            if category not in cats:
                continue
            scores = entry.get("scores")
            if not isinstance(scores, dict):
                raise ValueError(f"{name}: missing scores dictionary")
            flat = dict(scores)
            for key in ("min_score", "known_totals"):
                if key in entry:
                    flat[key] = entry[key]
        else:
            flat = dict(entry)
        flat["__name"] = name
        if flat.get("min_score") is not None:
            floor = numeric(flat["min_score"], f"{name}.min_score")
        else:
            floor = None
            if numeric(flat.get("known_totals"), f"{name}.known_totals") <= 0:
                raise ValueError(f"{name}: total must be positive")
        for model, value in flat.items():
            if model in BENCHMARK_META_KEYS or value is None:
                continue
            if not isinstance(model, str):
                raise ValueError(f"{name}: model identifier is not a string")
            score = numeric(value, f"{name}.{model}")
            if floor is None and not 0 <= score <= flat["known_totals"]:
                raise ValueError(f"{name}.{model}: rank outside normalization range")
        benchmarks.append(flat)
    return benchmarks


def benchmark_versions(text: str) -> dict[str, str]:
    """Read the documented SWE-Marathon chapter marker from its own header."""
    match = re.search(r"((?:#[^\n]*\n|\s*\n)*)\[SWE_Marathon\]", text)
    marathon = "v1.1" if match and "v1.1" in match[1] else "original"
    return {
        "HLE": "without tools",
        "ARC_AGI_2": "2",
        "CritPt": "original",
        "SWE_Marathon": marathon,
    }


def daily_snapshots(snapshots: list[Snapshot]) -> list[Snapshot]:
    """Select the final relevant observation per day in first-parent order."""
    days = {snapshot.date: snapshot for snapshot in snapshots}
    return [days[day] for day in sorted(days)]


def read_history(repo: Path) -> tuple[dict[str, list[Snapshot]], list[dict[str, str]]]:
    """Read all first-parent data changes, logging malformed category snapshots."""
    history: dict[str, list[Snapshot]] = {category: [] for category in CATEGORIES}
    skipped: list[dict[str, str]] = []
    previous: dict[str, str | None] = {}
    blobs: dict[str, str] = {}
    for line in git(
        repo, "log", "--first-parent", "--reverse", "--format=%H %cs"
    ).splitlines():
        commit, date = line.split()
        files: dict[str, str] = {}
        for entry in git(repo, "ls-tree", "-r", commit).splitlines():
            meta, entry_path = entry.split("\t", 1)
            if entry_path.endswith((".txt", ".toml")):
                files[entry_path] = meta.split()[2]
        for category in CATEGORIES:
            candidates = [
                "data/benchmarks.toml",
                "data/benchmarks.txt",
                f"data/{category}.txt",
                f"data/ranks_{category}.txt",
                f"ranks_{category}.txt",
            ]
            if category == "general":
                candidates.append("ranking_sample.txt")
            path = next((p for p in candidates if p in files), None)
            blob = files[path] if path else None
            if previous.get(category) == blob:
                continue
            previous[category] = blob
            if path is None or blob is None:
                continue
            if blob not in blobs:
                blobs[blob] = git(repo, "cat-file", "blob", blob)
            text = blobs[blob]
            try:
                benches = parse_benchmarks(text, path, category)
                if benches:
                    history[category].append(
                        Snapshot(date, commit, path, benches, benchmark_versions(text))
                    )
            except (ValueError, SyntaxError, TypeError) as exc:
                skipped.append(
                    {
                        "commit": commit,
                        "date": date,
                        "category": category,
                        "path": path,
                        "reason": str(exc),
                    }
                )
                # A broken final observation must not silently select an earlier same-day one.
                history[category] = [s for s in history[category] if s.date != date]
    return {c: daily_snapshots(items) for c, items in history.items()}, skipped


def collapse_states(states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse consecutive unchanged visible results while retaining source references."""
    result: list[dict[str, Any]] = []
    last_signature = ""
    for state in states:
        visible = {
            k: v
            for k, v in state.items()
            if k not in ("date", "commit", "sources", "added", "removed")
        }
        # Provenance changes alone should not make an otherwise identical frame.
        visible["models"] = [
            {
                k: v
                for k, v in m.items()
                if k not in ("observed_date", "observed_commit")
            }
            for m in state["models"]
        ]
        signature = json.dumps(visible, sort_keys=True)
        if result and signature == last_signature:
            result[-1]["sources"].append(
                {"date": state["date"], "commit": state["commit"]}
            )
        else:
            result.append(state)
        last_signature = signature
    # Keep the last observation date visible in the final hold, even without score changes.
    if result and states[-1]["date"] != result[-1]["date"]:
        result.append(states[-1])
    return result


def ranking_states(
    snapshots: list[Snapshot], *, collapse: bool = True
) -> list[dict[str, Any]]:
    """Apply today's ranking math to each snapshot's own normalization inputs."""
    states = []
    previous: set[str] = set()
    for snapshot in snapshots:
        results, _ = calculate_rankings(snapshot.benchmarks)
        if not results:
            continue
        names = [str(b["__name"]) for b in snapshot.benchmarks]
        state = {
            "date": snapshot.date,
            "commit": snapshot.commit,
            "version": "aggregate",
            "benchmarks": names,
            "added": sorted(set(names) - previous),
            "removed": sorted(previous - set(names)),
            "sources": [{"date": snapshot.date, "commit": snapshot.commit}],
            "models": [
                {
                    "name": name,
                    "value": 100 * (1 - score),
                    "composite": score,
                    "count": n,
                    "penalty": 0.25 if n == 1 else 0.10 if n == 2 else 0.0,
                }
                for name, score, _, _, n, _ in results
            ],
        }
        states.append(state)
        previous = set(names)
    return collapse_states(states) if collapse else states


def progress_states(
    snapshots: list[Snapshot], benchmark: str, *, collapse: bool = True
) -> list[dict[str, Any]]:
    """Retain latest past scores per version, including downward corrections."""
    retained: dict[str, dict[str, Any]] = {}
    states: list[dict[str, Any]] = []
    version = ""
    baseline: float | None = None
    for snapshot in snapshots:
        bench = next((b for b in snapshot.benchmarks if b["__name"] == benchmark), None)
        if bench is not None and bench.get("min_score") is None:
            # Early HLE data contains ranks, which are not benchmark percentages.
            continue
        current_version = snapshot.versions[benchmark] if bench is not None else version
        new_chapter = current_version != version
        if new_chapter:
            retained = {}
            baseline = None
            version = current_version
        observed = set()
        if bench is not None:
            for name, value in bench.items():
                if name in BENCHMARK_META_KEYS or value is None:
                    continue
                score = numeric(value, f"{benchmark}.{name}")
                if not 0 <= score <= 100:
                    raise ValueError(
                        f"{snapshot.commit}: {benchmark}.{name} outside 0–100"
                    )
                observed.add(name)
                retained[name] = {
                    "name": name,
                    "value": score,
                    "observed_date": snapshot.date,
                    "observed_commit": snapshot.commit,
                }
        if not retained:
            continue
        models = sorted(
            [dict(m, retained=name not in observed) for name, m in retained.items()],
            key=lambda m: (-m["value"], m["name"]),
        )
        best = float(models[0]["value"])
        if baseline is None:
            baseline = best
        states.append(
            {
                "date": snapshot.date,
                "commit": snapshot.commit,
                "version": version,
                "benchmark": benchmark,
                "baseline": baseline,
                "best": best,
                "gain": best - baseline,
                "models": models,
                "sources": [{"date": snapshot.date, "commit": snapshot.commit}],
            }
        )
    return collapse_states(states) if collapse else states


def snapshot_dict(snapshot: Snapshot) -> dict[str, Any]:
    """Serialize every normalization input for the companion manifest."""
    return asdict(snapshot)
