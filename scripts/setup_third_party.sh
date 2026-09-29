#!/usr/bin/env bash
# Official ResShift code at the commit this project was run with.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=third_party/ResShift
COMMIT=bb03b7d21614cace01787e097c8a6ab6b945227d
if [ ! -d "$REPO/.git" ]; then
  git clone https://github.com/zsyOAOA/ResShift.git "$REPO"
fi
git -C "$REPO" fetch --depth 1 origin "$COMMIT" 2>/dev/null || true
git -C "$REPO" checkout -q "$COMMIT"
echo "ResShift at $(git -C "$REPO" rev-parse --short HEAD)"
