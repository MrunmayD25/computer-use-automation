#!/usr/bin/env bash
# Print a draft and save an approved copy for replay.
#
#   scripts/review.sh <folder>
#
# Review the draft with computeruse review --capability before running this
# script. This script passes --approve and saves without another prompt.
# Risky actions still require approval during replay.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/sites.sh

FOLDER="${1:-}"
if [ ! -f "$FOLDER/capability.draft.json" ]; then
  echo "error: no capability.draft.json in that folder; run discover.sh first" >&2
  exit 2
fi
if [ -e "$FOLDER/capability.json" ]; then
  redo_or_stop "$FOLDER/capability.json" \
    "$FOLDER already holds an approved capability.json."
fi
logged "$FOLDER/review.log" uv run computeruse review \
  --capability "$FOLDER/capability.draft.json" \
  --approve --output "$FOLDER/capability.json"
