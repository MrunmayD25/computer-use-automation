#!/usr/bin/env bash
# Start a target application at its local address.
#
#   scripts/start.sh <responsive|components|canvas> [--reset]
#
# The first run installs the applications and seeds the database. --reset
# reseeds the database and removes accounts created by earlier runs.
#
# The applications require Node.js 25.9 or newer. If NODE is set, the script
# validates and uses it. Otherwise it uses a compatible node from PATH or
# downloads the release in scripts/sites.sh into .runtime/. Downloads must
# match the declared SHA-256.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/sites.sh
site "${1:-}"

resolve_node
if [ "${2:-}" = "--reset" ]; then
  prepare_sites reset
else
  prepare_sites
fi
echo "Serving ${APP} at ${WEBSITE}"
run uv run python -m evaluation.sites start --app "$APP" --node "$NODE"
