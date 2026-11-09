#!/bin/sh
# Downloads the pinned Pyodide core and the NumPy wheel into .cache/pyodide/,
# checks each file's SHA-256, and writes a lock file that lists NumPy only.
# cmd/build copies the result to public/papers/pyodide/.
#
#   scripts/fetch-pyodide.sh            # skips files that already match
set -eu

VERSION=0.29.5
BASE="https://cdn.jsdelivr.net/pyodide/v${VERSION}/full"
DEST="$(cd "$(dirname "$0")/.." && pwd)/.cache/pyodide/${VERSION}"
mkdir -p "$DEST"

# file  sha256
FILES="pyodide.js 7f832a350240263d9946a9c3c877f7bcdab6c37d6dc65f72cd3be5905dca62dd
pyodide.asm.js 356c42f69e1695397e9d8670bd3c2e678248cde76e18bdae1731e848537b47d7
pyodide.asm.wasm 54309a5a2cfd757b1f0fdbc9093c92503f7b341500110329d932487183912718
python_stdlib.zip 831fd1e535084b972f87c6a35275de3d236efd5166f5e77ce89497710635e73c
pyodide-lock.json 14d2c2dba101277999e17135e653d8f15389ad1437f53eae213bf0c3cdff723d
numpy-2.2.5-cp313-cp313-pyemscripten_2025_0_wasm32.whl 800c98edc0c864dfa49f07005680c699b4b42b84eae1f8cb19d35b3634e7f05c"

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

echo "$FILES" | while read -r name sum; do
  file="$DEST/upstream/$name"
  mkdir -p "$DEST/upstream"
  if [ ! -f "$file" ] || [ "$(sha256 "$file")" != "$sum" ]; then
    echo "fetch-pyodide: downloading $name"
    curl -fsSL --retry 3 -o "$file.tmp" "$BASE/$name"
    mv "$file.tmp" "$file"
  fi
  got=$(sha256 "$file")
  if [ "$got" != "$sum" ]; then
    echo "fetch-pyodide: $name has SHA-256 $got, want $sum" >&2
    exit 1
  fi
done

# The served directory: the core files, the NumPy wheel, and a lock file that
# lists NumPy only, so the browser can't be asked to fetch anything else.
OUT="$DEST/dist"
rm -rf "$OUT"
mkdir -p "$OUT"
for f in pyodide.js pyodide.asm.js pyodide.asm.wasm python_stdlib.zip numpy-2.2.5-cp313-cp313-pyemscripten_2025_0_wasm32.whl; do
  cp "$DEST/upstream/$f" "$OUT/$f"
done
python3 - "$DEST/upstream/pyodide-lock.json" "$OUT/pyodide-lock.json" <<'PY'
import json, sys
lock = json.load(open(sys.argv[1]))
lock["packages"] = {"numpy": lock["packages"]["numpy"]}
json.dump(lock, open(sys.argv[2], "w"), separators=(",", ":"))
PY
echo "fetch-pyodide: Pyodide ${VERSION} with NumPy is ready in $OUT"
