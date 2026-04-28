#!/bin/bash
# Backward-compatible wrapper around the Makefile plot refresh target.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -n "${WEBSITE_PLOTS_DIR:-}" ]]; then
  exec make refresh-plots WEBSITE_PLOTS_DIR="$WEBSITE_PLOTS_DIR"
fi

exec make refresh-plots
