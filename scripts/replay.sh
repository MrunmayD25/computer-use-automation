#!/usr/bin/env bash
# Replay an approved capability in a visible browser without a model.
#
#   scripts/replay.sh <responsive|components|canvas> <folder> NAME=VALUE ...
#
# Supply every input declared by the capability. The review command lists
# them. Each replay selects the next unused replay-N.jsonl journal name and
# writes intervention events to replay-N.human.jsonl. replay-N.log records
# structured events and placeholders without copying terminal values.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/sites.sh
site "${1:-}"
require_site

FOLDER="${2:-}"
if [ ! -f "$FOLDER/capability.json" ]; then
  echo "error: no approved capability.json in that folder; run review.sh first" >&2
  exit 2
fi
shift 2
inputs=()
for pair in "$@"; do
  inputs+=(--input "$pair")
done
number=1
while [ -e "$FOLDER/replay-$number.jsonl" ] || [ -e "$FOLDER/replay-$number.log" ]; do
  number=$((number + 1))
done
logged "$FOLDER/replay-$number.log" uv run computeruse replay --headed \
  --website "$WEBSITE" \
  --profile "$PROFILE" \
  --capability "$FOLDER/capability.json" \
  ${inputs[@]+"${inputs[@]}"} \
  --journal "$FOLDER/replay-$number.jsonl"
