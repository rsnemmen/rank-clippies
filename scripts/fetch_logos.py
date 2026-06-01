#!/usr/bin/env python3
"""Download and rasterize company logos into docs/assets/logos/.

Reads distinct ``company`` values from data/models.toml, fetches each
white SVG from Simple Icons (https://simpleicons.org), rasterizes it to a
128×128 transparent PNG, and writes docs/assets/logos/<company>.png.

Requirements:
    pip install cairosvg

Run once when adding a new company, then commit the resulting PNGs.  The
PNGs are the sole runtime asset; this script is not needed in production.

Usage:
    python scripts/fetch_logos.py
"""

from __future__ import annotations

import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

# ── paths (relative to this script's parent) ────────────────────────────────
_REPO_ROOT = Path(__file__).parent.parent
_MODELS_TOML = _REPO_ROOT / "data" / "models.toml"
_LOGO_DIR = _REPO_ROOT / "docs" / "assets" / "logos"

# ── Simple Icons slug overrides ──────────────────────────────────────────────
# Map our internal company slug to the Simple Icons slug when they differ.
# See https://simpleicons.org for available icons.
# If a slug is not available in Simple Icons, omit it from the override map;
# those companies will be skipped and models will render with plain markers.
_SLUG_OVERRIDE: dict[str, str] = {
    "xai": "x",             # xAI → X (Twitter/X icon on Simple Icons)
    "moonshot": "moonshotai",  # Moonshot AI (Kimi) → moonshotai
}

_LOGO_SIZE_PX = 128


def _collect_companies(models_file: Path) -> list[str]:
    """Return sorted list of unique company slugs from models.toml."""
    with models_file.open("rb") as f:
        data = tomllib.load(f)
    companies: set[str] = set()
    for meta in data.values():
        if isinstance(meta, dict) and meta.get("company"):
            companies.add(str(meta["company"]))
    return sorted(companies)


def _fetch_svg(slug: str) -> bytes:
    """Download the white SVG for a Simple Icons slug."""
    url = f"https://cdn.simpleicons.org/{slug}/white"
    req = urllib.request.Request(url, headers={"User-Agent": "fetch-logos/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
        return resp.read()


def _rasterize(svg_bytes: bytes, size: int) -> bytes:
    """Rasterize SVG bytes to a square transparent PNG at the given pixel size."""
    try:
        import cairosvg  # type: ignore[import]
    except ImportError:
        sys.exit(
            "Error: cairosvg is required.  Install with:  pip install cairosvg"
        )
    return cairosvg.svg2png(  # type: ignore[no-any-return]
        bytestring=svg_bytes,
        output_width=size,
        output_height=size,
        background_color="transparent",
    )


def main() -> None:
    companies = _collect_companies(_MODELS_TOML)
    print(f"Found {len(companies)} company slug(s): {', '.join(companies)}")

    _LOGO_DIR.mkdir(parents=True, exist_ok=True)

    ok: list[str] = []
    skipped: list[str] = []

    for company in companies:
        out_path = _LOGO_DIR / f"{company}.png"
        si_slug = _SLUG_OVERRIDE.get(company, company)

        print(f"  {company} (slug: {si_slug}) ...", end=" ", flush=True)
        try:
            svg_bytes = _fetch_svg(si_slug)
            png_bytes = _rasterize(svg_bytes, _LOGO_SIZE_PX)
            out_path.write_bytes(png_bytes)
            print(f"saved ({len(png_bytes):,} bytes)")
            ok.append(company)
        except urllib.error.HTTPError as exc:
            print(f"SKIPPED (HTTP {exc.code} — icon not found?)")
            skipped.append(company)
        except Exception as exc:  # noqa: BLE001
            print(f"SKIPPED ({exc})")
            skipped.append(company)

    print(f"\nDone: {len(ok)} saved, {len(skipped)} skipped.")
    if skipped:
        print(f"Skipped: {', '.join(skipped)}")
        print(
            "Models mapped to skipped companies will render without a logo (plain marker)."
        )


if __name__ == "__main__":
    main()
