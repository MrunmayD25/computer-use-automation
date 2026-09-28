# Shared helpers resolve site names, prepare applications, and log commands.
# Source this file; do not run it.

site() {
  case "${1:-}" in
    responsive)
      APP=white-label-responsive
      PORT=4363
      ;;
    components)
      APP=web-component-operations
      PORT=4360
      ;;
    canvas)
      APP=canvas-teller
      PORT=4390
      ;;
    *)
      echo "error: the site must be responsive, components, or canvas" >&2
      exit 2
      ;;
  esac
  WEBSITE="http://127.0.0.1:${PORT}/"
  PROFILE="evaluation/sites/profiles/${APP}.yaml"
  SITE="$1"
}

# If the site is unavailable, stop and print its start command.
require_site() {
  if ! curl -s -o /dev/null --max-time 5 "$WEBSITE"; then
    echo "error: the site is unavailable at $WEBSITE" >&2
    echo "Start the site in another terminal and leave it running:" >&2
    echo "  scripts/start.sh $SITE" >&2
    exit 2
  fi
}

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

# Let the CLI persist structured evidence without capturing terminal text.
logged() {
  local file="$1"
  shift
  PYTHONUNBUFFERED=1 "$@" --log "$file"
}

# Ask before deleting an existing run's files. Only an explicit yes at a
# terminal permits deletion. Any other answer, or no terminal, stops the run.
redo_or_stop() {
  local path="$1" error="$2" answer
  if [ ! -t 0 ]; then
    echo "error: $error" >&2
    exit 2
  fi
  printf '%s\nRedo it? This deletes the old files. [y/N] ' "$error"
  read -r answer
  case "$answer" in
    y | Y | yes | YES)
      rm -rf "$path"
      echo "Deleted the old files."
      ;;
    *)
      echo "Stopped. No files were changed." >&2
      exit 2
      ;;
  esac
}

# uv loads the model key from .env when the file exists.
uv_env() {
  if [ -f .env ]; then
    UV_RUN=(uv run --env-file .env)
  else
    UV_RUN=(uv run)
  fi
}

# The target applications need Node.js 25.9 or newer.
NODE_RELEASE=v26.10.0

new_enough() {
  local version major minor
  version="$("$1" --version 2>/dev/null)" || return 1
  version="${version#v}"
  major="${version%%.*}"
  minor="${version#*.}"
  minor="${minor%%.*}"
  [ "$major" -gt 25 ] || { [ "$major" -eq 25 ] && [ "$minor" -ge 9 ]; }
}

release_node() {
  local platform sum archive folder
  case "$(uname -s)-$(uname -m)" in
    Darwin-arm64)
      platform=darwin-arm64
      sum=751fdf7439f115d87ee2a8f3f18c065b6151852068e3e666ac60ac2996f75ac9
      ;;
    Darwin-x86_64)
      platform=darwin-x64
      sum=ebbe9ab9b58ad6bb54390d6e2c862c1afa7d4475fb7e8ae8146acde211bf70df
      ;;
    Linux-aarch64 | Linux-arm64)
      platform=linux-arm64
      sum=423a41bff8e2a2fa15e702fefe2919ef95823b2378744daccb8439302534b44f
      ;;
    Linux-x86_64)
      platform=linux-x64
      sum=cb5c9ce9c80d7b8821e3a258543c71b939138cf17c74d5cc44bbe85d6dbc5ad8
      ;;
    *)
      echo "error: no Node.js release for this machine; set NODE to Node.js 25.9 or newer" >&2
      exit 2
      ;;
  esac
  folder=".runtime/node-${NODE_RELEASE}-${platform}"
  if [ ! -x "$folder/bin/node" ]; then
    archive=".runtime/node-${NODE_RELEASE}-${platform}.tar.gz"
    mkdir -p .runtime
    echo "Downloading Node.js ${NODE_RELEASE} into .runtime/"
    run curl -fsSL -o "$archive" \
      "https://nodejs.org/dist/${NODE_RELEASE}/node-${NODE_RELEASE}-${platform}.tar.gz"
    if [ "$(shasum -a 256 "$archive" | cut -d ' ' -f 1)" != "$sum" ]; then
      rm -f "$archive"
      echo "error: the Node.js download does not match its SHA-256" >&2
      exit 2
    fi
    tar -xzf "$archive" -C .runtime
    rm -f "$archive"
  fi
  NODE="$PWD/$folder/bin/node"
}

# Validate an explicit NODE first. Otherwise use a compatible node from PATH
# or download and verify the release above. Require Node.js 25.9 or newer.
resolve_node() {
  if [ -z "${NODE:-}" ]; then
    if command -v node >/dev/null && new_enough node; then
      NODE="$(command -v node)"
    else
      release_node
    fi
  fi
  if ! new_enough "$NODE"; then
    echo "error: NODE must be Node.js 25.9 or newer" >&2
    exit 2
  fi
}

# Install unbuilt applications. Seed the database if no seed exists or reset
# was requested. Call resolve_node first to set NODE.
prepare_sites() {
  local built="evaluation/runtime/demo-sources/environments/database/financial/dist/services.js"
  if [ ! -f "$built" ]; then
    run uv run python -m evaluation.sites install --node "$NODE"
  fi
  if [ ! -f .runtime/evaluation.seed.sqlite ] || [ "${1:-}" = reset ]; then
    run uv run python -m evaluation.sites seed --node "$NODE"
  fi
}
