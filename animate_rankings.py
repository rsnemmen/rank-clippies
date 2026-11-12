#!/usr/bin/env python
"""Animate repository observations as relative rankings or fixed-scale benchmark progress.

Usage: python animate_rankings.py all --view both
       python animate_rankings.py general --view progress --benchmark HLE
"""

import argparse
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
from typing import Any

from ranking_history import (
    BENCHMARK_LABELS,
    CATEGORIES,
    DEFAULT_BENCHMARKS,
    git,
    collapse_states,
    progress_states,
    ranking_states,
    read_history,
    snapshot_dict,
)

PALETTE = (
    "#67c9eb",
    "#f2b75e",
    "#80d4aa",
    "#c5a1ef",
    "#f38993",
    "#90aff4",
    "#d4d27b",
    "#e7a9cd",
    "#6dd4cb",
    "#b9c8dc",
)
BACKGROUND = "#101b2b"
FOREGROUND = "#edf3fa"
MUTED = "#a9bacf"


def model_color(name: str) -> str:
    """Assign a stable color independent of category, history, or Python hash seed."""
    return PALETTE[
        int.from_bytes(hashlib.sha256(name.encode()).digest()[:4]) % len(PALETTE)
    ]


def frame_position(
    frame: int, frames: int, count: int, fps: int
) -> tuple[int, int, float]:
    """Allocate opening/final holds and eased transitions with a pause at each snapshot."""
    if count == 1:
        return 0, 0, 1.0
    opening = min(2 * fps, frames * 0.1)
    closing = min(3 * fps, frames * 0.15)
    position = (frame - opening) / max(1, frames - 1 - opening - closing) * (count - 1)
    if position <= 0:
        return 0, 0, 1.0
    if position >= count - 1:
        return count - 1, count - 1, 1.0
    previous = int(position)
    fraction = min(1.0, (position - previous) / 0.6)
    eased = fraction * fraction * (3 - 2 * fraction)
    return previous, previous + 1, eased


def interpolate_models(
    previous: dict[str, Any], current: dict[str, Any], fraction: float, top_n: int
) -> list[dict[str, Any]]:
    """Move bars smoothly; fade entrants at their first recorded value, never from zero."""
    before = {m["name"]: (i, m) for i, m in enumerate(previous["models"][:top_n])}
    after = {m["name"]: (i, m) for i, m in enumerate(current["models"][:top_n])}
    chapter_change = previous["version"] != current["version"]
    if chapter_change:
        source = before if fraction < 0.5 else after
        alpha = 1 - 2 * fraction if fraction < 0.5 else 2 * fraction - 1
        return [dict(m, position=i, alpha=alpha) for i, m in source.values()]
    rows = []
    for name in dict.fromkeys([*before, *after]):
        old_index, old = before.get(name, after.get(name, (0, {})))
        new_index, new = after.get(name, before.get(name, (0, {})))
        alpha = (
            fraction if name not in before else 1 - fraction if name not in after else 1
        )
        rows.append(
            dict(
                new,
                position=old_index + (new_index - old_index) * fraction,
                value=old["value"] + (new["value"] - old["value"]) * fraction,
                alpha=alpha,
            )
        )
    return rows


def render_video(
    states: list[dict[str, Any]],
    category: str,
    view: str,
    benchmark: str,
    output: Path,
    duration: float,
    fps: int,
    top_n: int,
    highlight: str | None,
) -> None:
    """Stream 1080p frames through Matplotlib's FFmpeg writer with bounded memory."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        from matplotlib.animation import FFMpegWriter
    except ImportError as exc:
        raise ValueError(
            "Animation requires matplotlib: pip install matplotlib"
        ) from exc
    if not FFMpegWriter.isAvailable():
        raise ValueError("Animation requires ffmpeg on PATH")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "text.color": FOREGROUND,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "savefig.bbox": None,
        }
    )
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=BACKGROUND)
    is_progress = view == "progress"
    bottom, height = (0.37, 0.43) if is_progress else (0.25, 0.55)
    ax = fig.add_axes((0.24, bottom, 0.63, height), facecolor=BACKGROUND)
    ax.set_xlim(0, 100)
    ax.set_ylim(top_n - 0.3, -0.8)
    ax.set_yticks([])
    ax.set_xticks(range(0, 101, 20))
    ax.grid(axis="x", color="#314156", linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel(
        "Benchmark score (%)" if is_progress else "Relative ranking score", labelpad=12
    )
    fig.text(
        0.055,
        0.947,
        "RANK CLIPPIES  /  REPOSITORY HISTORY",
        fontsize=14,
        color="#67c9eb",
        weight="bold",
    )
    title = f"{category.title()} · {'benchmark progress' if is_progress else 'ranking race'}"
    fig.text(0.055, 0.889, title, fontsize=32, weight="bold")
    subtitle = fig.text(0.055, 0.844, "", fontsize=18, color=MUTED)
    date_label = fig.text(0.95, 0.914, "", fontsize=29, ha="right", weight="bold")
    fig.text(
        0.95, 0.874, "Repository observation", fontsize=13, ha="right", color=MUTED
    )
    fig.text(
        0.91,
        bottom + height + 0.012,
        "SCORE" if is_progress else "SCORE / BENCHES",
        fontsize=11,
        ha="center",
        color=MUTED,
    )
    detail = fig.text(0.055, 0.077, "", fontsize=13, color=MUTED, linespacing=1.6)
    fig.text(
        0.055,
        0.025,
        "Dates are repository observations, not release dates.  •  Exact model identifiers; no alias merging.",
        fontsize=12,
        color=MUTED,
    )
    event = fig.text(
        0.055, 0.17 if not is_progress else 0.285, "", fontsize=14, color="#f2b75e"
    )
    status = fig.text(0.95, 0.025, "", ha="right", fontsize=11, color=MUTED)
    artists: dict[str, tuple[Any, Any, Any]] = {}
    for name in sorted({m["name"] for s in states for m in s["models"]}):
        bar = ax.barh(
            0,
            0,
            height=0.66,
            color=model_color(name),
            edgecolor=FOREGROUND if name == highlight else "none",
            linewidth=2.5,
        )[0]
        label = ax.text(-2, 0, "", ha="right", va="center", fontsize=15, clip_on=False)
        value = ax.text(103, 0, "", ha="left", va="center", fontsize=15, clip_on=False)
        artists[name] = bar, label, value

    timeline: Any = None
    best_line: Any = None
    highlight_line: Any = None
    cursor: Any = None
    dates = [
        float(date.fromisoformat(s["date"]).toordinal() - date(1970, 1, 1).toordinal())
        for s in states
    ]
    if is_progress:
        timeline = fig.add_axes((0.24, 0.145, 0.63, 0.10), facecolor=BACKGROUND)
        timeline.set_xlim(dates[0] - 1, dates[-1] + 1)
        timeline.set_ylim(0, 100)
        timeline.set_yticks([0, 50, 100])
        timeline.tick_params(labelsize=11)
        timeline.set_ylabel("Best known", fontsize=12)
        locator = mdates.AutoDateLocator(minticks=3, maxticks=6)  # type: ignore[no-untyped-call]
        timeline.xaxis.set_major_locator(locator)
        timeline.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))  # type: ignore[no-untyped-call]
        timeline.grid(color="#314156", linewidth=0.5)
        for spine in timeline.spines.values():
            spine.set_visible(False)
        (best_line,) = timeline.plot(
            [],
            [],
            color="#67c9eb",
            linewidth=2.5,
            drawstyle="steps-post",
            marker="o",
            markersize=3,
        )
        (highlight_line,) = timeline.plot(
            [],
            [],
            color="#f2b75e",
            linewidth=2,
            drawstyle="steps-post",
            marker="o",
            markersize=3,
        )
        cursor = timeline.axvline(dates[0], color=MUTED, linewidth=1)

    frames = round(duration * fps)
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=-1,
        extra_args=[
            "-crf",
            "20",
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-threads",
            "2",
        ],
        metadata={
            "title": title,
            "comment": "Repository observations; see JSON manifest",
        },
    )
    previews = {0: "opening", frames // 2: "middle", frames - 1: "final"}
    chapter_previews: set[str] = set()
    try:
        with writer.saving(fig, str(output), dpi=100):
            for frame in range(frames):
                old, new, fraction = frame_position(frame, frames, len(states), fps)
                current = states[new]
                # The target observation date is shown as soon as its transition starts.
                date_label.set_text(current["date"])
                subtitle.set_text(
                    f"{BENCHMARK_LABELS[benchmark]}"
                    + (
                        f"  /  {current['version']}"
                        if benchmark == "SWE_Marathon"
                        else ""
                    )
                    + "  /  fixed 0–100 scale"
                    if is_progress
                    else "100 × (1 − composite score)  /  higher is better  /  fixed 0–100 scale"
                )
                for group in artists.values():
                    for artist in group:
                        artist.set_visible(False)
                rows = interpolate_models(states[old], current, fraction, top_n)
                # Temporarily omit colliding labels as bars cross; all return at the hold.
                labelled_positions: list[float] = []
                for row in sorted(
                    rows,
                    key=lambda r: (
                        r["name"] != highlight,
                        -r["alpha"],
                        -r["value"],
                        r["name"],
                    ),
                ):
                    bar, label, value = artists[row["name"]]
                    alpha = row["alpha"] * (
                        0.4 if highlight and row["name"] != highlight else 1
                    )
                    for artist in (bar, label, value):
                        artist.set_visible(alpha > 0)
                        artist.set_alpha(alpha)
                    y = row["position"]
                    has_space = all(
                        abs(y - position) >= 0.55 for position in labelled_positions
                    )
                    label.set_visible(alpha > 0 and has_space)
                    value.set_visible(alpha > 0 and has_space)
                    if alpha > 0 and has_space:
                        labelled_positions.append(y)
                    bar.set_zorder(2 + row["alpha"] + row["value"] / 1000)
                    bar.set_y(y - 0.33)
                    bar.set_width(row["value"])
                    label.set_position((-2, y))
                    marker = (
                        " *"
                        if row.get("retained")
                        else " †"
                        if row.get("sparse")
                        else ""
                    )
                    label.set_text(row["name"] + marker)
                    value.set_position((103, y))
                    value.set_text(
                        f"{row['value']:.1f}"
                        if is_progress
                        else f"{row['value']:.1f}  /  {row['count']}{' †' if row.get('sparse') else ''}"
                    )
                if is_progress:
                    chapter_start = new
                    while (
                        chapter_start
                        and states[chapter_start - 1]["version"] == current["version"]
                    ):
                        chapter_start -= 1
                    shown = states[chapter_start : new + 1]
                    shown_dates = dates[chapter_start : new + 1]
                    best_line.set_data(shown_dates, [s["best"] for s in shown])
                    cursor.set_xdata([dates[new], dates[new]])
                    if highlight:
                        highlight_line.set_data(
                            shown_dates,
                            [
                                next(
                                    (
                                        m["value"]
                                        for m in s["models"]
                                        if m["name"] == highlight
                                    ),
                                    math.nan,
                                )
                                for s in shown
                            ],
                        )
                    event.set_text(
                        f"Best known: {current['best']:.1f}%    |    {current['gain']:+.1f} benchmark points since chapter's initial leader"
                        + (
                            "    |    NEW VERSION · comparison reset"
                            if states[old]["version"] != current["version"]
                            else ""
                        )
                    )
                    detail.set_text(
                        "* Retained historical entry · latest observation in this benchmark version.  Corrections may lower scores.\n"
                        + (f"Gold timeline: {highlight}.  " if highlight else "")
                        + "Raw benchmark performance; comparisons reset at version changes."
                    )
                else:
                    changes = []
                    if current["added"]:
                        changes.append("Added: " + ", ".join(current["added"]))
                    if current["removed"]:
                        changes.append("Retired: " + ", ".join(current["removed"]))
                    change_text = "  |  ".join(changes) or "Benchmark set unchanged"
                    event.set_text(textwrap.fill(change_text, 145))
                    detail.set_text(
                        textwrap.fill(
                            "Benchmarks: " + ", ".join(current["benchmarks"]), 160
                        )
                        + "\n† Fewer than 3 benchmarks.  Scores are shrunk toward a below-average result in proportion to how few benchmarks a model has.  Relative standing depends on the observed benchmark set."
                    )
                status.set_text(
                    "Transition · interpolated bars"
                    if old != new and fraction < 1
                    else f"{new + 1} / {len(states)} observations"
                )
                writer.grab_frame(facecolor=BACKGROUND)
                if frame in previews:
                    fig.savefig(
                        output.with_name(f"{output.stem}_{previews[frame]}.png"),
                        dpi=100,
                    )
                if (
                    is_progress
                    and fraction == 1
                    and current["version"] not in chapter_previews
                ):
                    chapter_previews.add(current["version"])
                    fig.savefig(
                        output.with_name(
                            f"{output.stem}_chapter{len(chapter_previews)}.png"
                        ),
                        dpi=100,
                    )
                if frame % max(1, fps * 10) == 0:
                    print(
                        f"  {output.name}: {frame / fps:.0f}/{duration:g} seconds",
                        flush=True,
                    )
    finally:
        plt.close(fig)


def make_gif(video: Path) -> None:
    """Convert the MP4 to a smaller GIF with FFmpeg, avoiding frame accumulation in Python."""
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(video),
            "-filter_complex",
            "fps=12,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse",
            "-loop",
            "0",
            str(video.with_suffix(".gif")),
        ],
        check=True,
    )


def parse_args() -> argparse.Namespace:
    """Validate animation CLI options before loading history or rendering."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("category", choices=("all", *CATEGORIES))
    parser.add_argument(
        "--view", choices=("ranking", "progress", "both"), default="both"
    )
    parser.add_argument(
        "--benchmark", choices=(*BENCHMARK_LABELS, "SWE-Marathon", "ARC-AGI-2")
    )
    parser.add_argument("--start-date", type=date.fromisoformat)
    parser.add_argument("--end-date", type=date.fromisoformat)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--highlight", help="Exact model identifier; also draw its progress trajectory"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures/animations"))
    parser.add_argument(
        "--gif", action="store_true", help="Also export a 960px, 12 FPS GIF"
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Reconstruct and export data without rendering",
    )
    args = parser.parse_args()
    if args.benchmark and (args.category == "all" or args.view == "ranking"):
        parser.error("--benchmark requires a single category and a progress view")
    if args.top_n < 1 or args.top_n > 20:
        parser.error("--top-n must be between 1 and 20 for readable 1080p output")
    if (
        args.fps < 1
        or not math.isfinite(args.duration)
        or args.duration <= 0
        or args.duration * args.fps < 2
    ):
        parser.error(
            "--fps and --duration must be positive and produce at least two frames"
        )
    if args.start_date and args.end_date and args.start_date > args.end_date:
        parser.error("--start-date must not be later than --end-date")
    args.benchmark = {"SWE-Marathon": "SWE_Marathon", "ARC-AGI-2": "ARC_AGI_2"}.get(
        args.benchmark, args.benchmark
    )
    return args


def main() -> None:
    """Build auditable manifests and render the requested category/view combinations."""
    args = parse_args()
    repo = Path(__file__).resolve().parent
    try:
        history, skipped = read_history(repo)
        categories = CATEGORIES if args.category == "all" else (args.category,)
        views = ("ranking", "progress") if args.view == "both" else (args.view,)
        jobs = []
        for category in categories:
            benchmark = args.benchmark or DEFAULT_BENCHMARKS[category]
            for view in views:
                states = (
                    ranking_states(history[category], collapse=False)
                    if view == "ranking"
                    else progress_states(history[category], benchmark, collapse=False)
                )
                states = [
                    s
                    for s in states
                    if (not args.start_date or s["date"] >= args.start_date.isoformat())
                    and (not args.end_date or s["date"] <= args.end_date.isoformat())
                ]
                states = collapse_states(states)
                if not states:
                    raise ValueError(
                        f"No usable {view} observations for {category} in the requested interval"
                    )
                if args.highlight and not any(
                    m["name"] == args.highlight for s in states for m in s["models"]
                ):
                    raise ValueError(
                        f"Highlight model {args.highlight!r} absent from {category} {view}"
                    )
                jobs.append((category, view, benchmark, states))
        if not args.manifest_only and shutil.which("ffmpeg") is None:
            raise ValueError("Animation requires ffmpeg on PATH")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for category, view, benchmark, states in jobs:
            suffix = f"_{benchmark}" if view == "progress" else ""
            output = args.output_dir / f"{category}_{view}{suffix}.mp4"
            manifest: dict[str, Any] = {
                "schema_version": 1,
                "head": git(repo, "rev-parse", "HEAD").strip(),
                "category": category,
                "view": view,
                "benchmark": benchmark if view == "progress" else None,
                "date_semantics": "committer calendar date; final relevant snapshot per day; first-parent history",
                "methodology": "Current median normalization with coverage shrinkage; progress retains latest past scores per version, not running maxima.",
                "settings": {
                    "width": 1920,
                    "height": 1080,
                    "fps": args.fps,
                    "duration": round(args.duration * args.fps) / args.fps,
                    "frames": round(args.duration * args.fps),
                    "top_n": args.top_n,
                    "highlight": args.highlight,
                    "gif": args.gif,
                    "start_date": str(args.start_date) if args.start_date else None,
                    "end_date": str(args.end_date) if args.end_date else None,
                    "timing": "Equal time per changed snapshot; 2s opening/3s final holds (scaled for short videos); 60% transition/40% pause",
                    "axis_limits": [0, 100],
                },
                "observations": [
                    snapshot_dict(s)
                    for s in history[category]
                    if not args.end_date or s.date <= args.end_date.isoformat()
                ],
                "states": states,
                "skipped_snapshots": [s for s in skipped if s["category"] == category],
                "rendered": False,
            }
            manifest_path = output.with_suffix(".json")
            manifest_path.write_text(
                json.dumps(manifest, indent=2, allow_nan=False) + "\n"
            )
            print(
                f"{category} {view}: {len(states)} states, {states[0]['date']} to {states[-1]['date']}; "
                f"{len(manifest['skipped_snapshots'])} skipped snapshots",
                flush=True,
            )
            if not args.manifest_only:
                render_video(
                    states,
                    category,
                    view,
                    benchmark,
                    output,
                    args.duration,
                    args.fps,
                    args.top_n,
                    args.highlight,
                )
                if args.gif:
                    make_gif(output)
                manifest["rendered"] = True
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, allow_nan=False) + "\n"
                )
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        sys.exit(f"Error: {exc}")


if __name__ == "__main__":
    main()
