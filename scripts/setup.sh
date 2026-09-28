#!/usr/bin/env bash
# Install the demo's dependencies and report their status.
#
#   scripts/setup.sh
#
# Existing installations and .env values are preserved. The script reports
# whether the model key is set without printing it.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/sites.sh

step() {
  printf '\n== %s\n' "$1"
}

step "uv"
if ! command -v uv >/dev/null; then
  echo "error: uv is not installed; see https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 2
fi
uv --version

step "Python packages"
run uv sync --locked

step "Chromium for Playwright"
run uv run playwright install chromium

step "Tk for the control window"
if uv run python -c "import tkinter; tkinter.Tcl()" 2>/dev/null; then
  tk="ready"
  echo "Tk is available."
else
  tk="missing: the control window opens in the browser"
  echo "Tk is missing. The control window opens in the browser."
fi

step "Model key"
if [ ! -f .env ]; then
  run cp .env.example .env
fi
if uv run --env-file .env python -c "import os, sys; sys.exit(0 if os.environ.get('OPENAI_API_KEY') else 1)"; then
  key="set in .env"
  echo "OPENAI_API_KEY is set in .env."
else
  key="missing: set OPENAI_API_KEY in .env before discovery"
  echo "OPENAI_API_KEY is empty. Set it in .env for discovery. Replay needs no key."
fi

step "Node.js for the target applications"
resolve_node
echo "Using $("$NODE" --version) at $NODE"

step "Target applications and database"
prepare_sites

step "Command check"
uv run computeruse --help >/dev/null
echo "computeruse runs."

step "Ready"
echo "  Python packages and computeruse: ready"
echo "  Chromium: ready"
echo "  Tk: $tk"
echo "  Node.js: $("$NODE" --version)"
echo "  Target applications and database: ready"
echo "  Model key: $key"
echo
echo "Start the site: scripts/start.sh responsive"
echo "In another terminal, run: scripts/discover.sh responsive lookup runs/demo"
