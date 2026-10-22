#!/usr/bin/env bash
# Every notebook header and pyproject.toml pins an exact released milume (milume==X.Y.Z):
# no ranges, no bare "milume", no preview wheels (milume @ <url>) or reinstall-package.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
files=(*.py pyproject.toml)
bad=$(grep -HnE '"milume([^"a-zA-Z0-9_-][^"]*)?"|milume @|reinstall-package' "${files[@]}" \
  | grep -vE '"milume==[0-9]+\.[0-9]+\.[0-9]+"' || true)
if [[ -n "$bad" ]]; then
  echo "milume must be pinned to an exact release (\"milume==X.Y.Z\"):" >&2
  echo "$bad" >&2
  exit 1
fi
echo "milume pins ok: $(grep -hoE '"milume==[0-9.]+"' "${files[@]}" | sort -u | tr '\n' ' ')"
