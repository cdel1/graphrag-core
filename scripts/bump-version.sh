#!/usr/bin/env bash
# Bump the graphrag-core version.
#
# The version lives in exactly ONE place — src/graphrag_core/__init__.py
# (__version__). pyproject.toml reads it dynamically via hatchling, and this
# script re-runs `uv lock` so uv.lock's editable self-entry never lags a bump
# (the drift that used to leave a trailing "sync uv.lock" commit after releases).
#
# Usage: scripts/bump-version.sh <version>       e.g. scripts/bump-version.sh 0.16.0
set -euo pipefail

NEW="${1:-}"
if [[ ! "$NEW" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][0-9A-Za-z.]+)?$ ]]; then
  echo "usage: $0 <version>   (semver, e.g. 0.16.0)" >&2
  exit 1
fi

# Resolve repo root so the script works from anywhere.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INIT="$ROOT/src/graphrag_core/__init__.py"

# Rewrite the single __version__ line (no sed/awk — a small Python rewrite).
python3 - "$NEW" "$INIT" <<'PYEOF'
import re, sys
new, path = sys.argv[1], sys.argv[2]
src = open(path).read()
src2, n = re.subn(
    r'^__version__\s*=\s*["\'][^"\']*["\']',
    f'__version__ = "{new}"',
    src, count=1, flags=re.M,
)
if n != 1:
    raise SystemExit(f"no __version__ line found in {path}")
open(path, "w").write(src2)
PYEOF

# Keep uv.lock in step with the new version.
( cd "$ROOT" && uv lock )

echo "Bumped __version__ -> $NEW and re-locked uv.lock."
echo "Next:"
echo "  git add -A && git commit -m \"chore(release): $NEW\""
echo "  git tag \"v$NEW\" && git push && git push --tags"
