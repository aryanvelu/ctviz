#!/usr/bin/env bash
# Run questions from examples/test_questions.txt through the full pipeline (no server needed) and save the
# exact public JSON a client would receive to runs/manual/<n>.json, plus a one-line verdict per question.
#   scripts/run_questions.sh              # all questions
#   scripts/run_questions.sh 3 17         # only questions 3 and 17
#   QUESTIONS=my_list.txt scripts/run_questions.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FILE="${QUESTIONS:-$ROOT/examples/test_questions.txt}"
OUT="$ROOT/runs/manual"; mkdir -p "$OUT"
PY="$ROOT/.venv/bin/python"
n=0
grep -v '^\s*#' "$FILE" | grep -v '^\s*$' | while IFS= read -r line; do
  n=$((n + 1))
  if [ $# -gt 0 ] && ! printf ' %s ' "$*" | grep -q " $n "; then continue; fi
  q="${line%% || *}"; flags=""; [ "$q" != "$line" ] && flags="${line#* || }"
  # shellcheck disable=SC2086
  "$ROOT/.venv/bin/ctviz" ask "$q" $flags --out "$OUT/$n.json" --quiet 2>/dev/null
  "$PY" - "$OUT/$n.json" "$n" "$q" <<'PYEOF'
import json, sys
path, n, q = sys.argv[1:4]
try:
    r = json.load(open(path))
except Exception as exc:
    print(f"[{n:>2}] ERROR {exc}  | {q}"); sys.exit()
m = r["meta"]; v = r.get("visualization") or {}
val = (m.get("validation") or {})
chk = "" if r["status"] != "ok" else (" valid" if val.get("passed") else " INVALID:" + ",".join(
    c["name"] for c in val.get("checks", []) if not c["passed"]))
what = v.get("type") or (r.get("clarification") or {}).get("reason")
print(f"[{n:>2}] {r['status']:<19} {what:<22} {m.get('coverage', {}).get('counting_method', ''):<15}{chk} | {q}")
print(f"      {(r.get('answer_summary') or (r.get('clarification') or {}).get('message', ''))[:220]}")
PYEOF
done
echo "Public JSON for each question: $OUT/<n>.json"
