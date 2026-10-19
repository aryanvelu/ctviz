#!/usr/bin/env bash
# Fails if .env is not ignored or if anything staged looks like a secret.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
git check-ignore -q .env || { echo "ERROR: .env is not gitignored"; exit 1; }
prefix="sk"; pat="${prefix}-(proj-)?[A-Za-z0-9_-]{20,}"
if git grep --cached -nE "$pat" -- . ':!scripts/secret_scan.sh'; then
  echo "ERROR: possible API key staged"; exit 1
fi
echo "secret scan: clean"
