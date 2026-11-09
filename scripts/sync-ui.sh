#!/usr/bin/env bash
# Copies the design system in ui/ into a course repo at frontend/src/ui/.
#
# Usage: scripts/sync-ui.sh REPO_DIR
#
# Each copied file gets a "generated, do not edit" header that names the
# source commit. Edit ui/ in weights-and-wires/sites, then run this script
# again in every course repo. The script prints a diff stat of the result.
set -euo pipefail

if [[ $# -ne 1 || ! -d "$1" ]]; then
  echo "usage: $0 REPO_DIR" >&2
  exit 2
fi

src="$(cd "$(dirname "$0")/.." && pwd)/ui"
repo="$(cd "$1" && pwd)"
dest="$repo/frontend/src/ui"
rev="$(git -C "$src" rev-parse --short HEAD 2>/dev/null || echo unknown)"
note="Generated from weights-and-wires/sites ui/ at $rev by scripts/sync-ui.sh. Do not edit here."

[[ -d "$repo/frontend/src" ]] || { echo "error: $repo/frontend/src does not exist" >&2; exit 1; }
mkdir -p "$dest"

for file in "$src"/*; do
  name="$(basename "$file")"
  case "$name" in
    *.css)     header="/* $note */" ;;
    *.js|*.ts) header="// $note" ;;
    *.md)      header="<!-- $note -->" ;;
    *)         continue ;;
  esac
  { printf '%s\n\n' "$header"; cat "$file"; } > "$dest/$name"
done

# Drop files that no longer exist in ui/.
for file in "$dest"/*; do
  [[ -e "$src/$(basename "$file")" ]] || rm -f "$file"
done

git -C "$repo" add -N "$dest" 2>/dev/null || true
git -C "$repo" diff --stat -- "$dest" || true
