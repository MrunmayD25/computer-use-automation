#!/usr/bin/env bash
# Discover a workflow in a visible browser and save a draft capability.
#
#   scripts/discover.sh <responsive|components|canvas> <lookup|write> <folder> [goal]
#
# The default goal comes from evaluation/sites/goals.yaml. The model proposes
# inputs from the goal, and the control window asks for missing values. Press
# Start to begin. If the output folder contains files, the script asks before
# deleting them. The run saves a draft, journals, and discovery.log. Approving
# the final review also saves capability.json for replay.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/sites.sh
site "${1:-}"
require_site

case "${2:-}" in
  lookup)
    TASK=member_status
    CONTRACT=evaluation/sites/member-status.contract.json
    EXTRA=()
    ;;
  write)
    TASK=open_account
    CONTRACT=evaluation/sites/open-account.contract.json
    # Lookup demonstrates the missing-member branch. Skip outcome discovery
    # for writes to avoid another model run for the same search failure.
    EXTRA=(--no-outcome-checks)
    ;;
  *)
    echo "error: the task must be lookup or write" >&2
    exit 2
    ;;
esac
FOLDER="${3:-}"
if [ -z "$FOLDER" ]; then
  echo "error: name a folder for the run, such as runs/demo" >&2
  exit 2
fi
# Journals and drafts cannot overwrite existing files, including files from
# an incomplete run. Reusing a folder requires explicit deletion approval.
if [ -d "$FOLDER" ] && [ -n "$(ls -A "$FOLDER")" ]; then
  redo_or_stop "$FOLDER" "$FOLDER already holds files from an earlier run."
fi
GOAL="${4:-}"
if [ -z "$GOAL" ]; then
  GOAL="$(uv run python -c 'import sys, yaml
goals = yaml.safe_load(open("evaluation/sites/goals.yaml"))
print(goals[sys.argv[1]][sys.argv[2]])' "$TASK" "$APP")"
fi

mkdir -p "$FOLDER"
uv_env
logged "$FOLDER/discovery.log" "${UV_RUN[@]}" computeruse discover --headed \
  --website "$WEBSITE" \
  --profile "$PROFILE" \
  --goal "$GOAL" \
  --contract "$CONTRACT" \
  ${EXTRA[@]+"${EXTRA[@]}"} \
  --capability-id "$TASK" \
  --save-capability "$FOLDER/capability.draft.json" \
  --save-approved "$FOLDER/capability.json" \
  --journal "$FOLDER/discovery.jsonl"
