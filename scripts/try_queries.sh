#!/usr/bin/env bash
# Smoke-test a running ctviz server without the frontend.
#   scripts/try_queries.sh                 # run the built-in query set against localhost:8000
#   scripts/try_queries.sh "your question" # run one query
#   BASE=http://host:port scripts/try_queries.sh
set -uo pipefail
BASE="${BASE:-http://localhost:8000}"
PY="$(dirname "$0")/../.venv/bin/python"; [ -x "$PY" ] || PY=python3

if ! curl -sf "$BASE/v1/health" >/dev/null; then
  echo "Server not reachable at $BASE. Start it with: .venv/bin/ctviz serve --port 8000"; exit 1
fi

run() {  # $1 = JSON body
  local t0 t1 out
  t0=$("$PY" -c 'import time;print(time.time())')
  out=$(curl -s -w '\n%{http_code}' -X POST "$BASE/v1/visualize" -H 'content-type: application/json' -d "$1")
  t1=$("$PY" -c 'import time;print(time.time())')
  printf '%s' "$out" | "$PY" -c '
import json, sys
raw = sys.stdin.read(); body, _, code = raw.rpartition("\n")
req = json.loads(sys.argv[1]); secs = float(sys.argv[3]) - float(sys.argv[2])
print("=" * 100); print(f"Q: {req[\"query\"]}  {({k: v for k, v in req.items() if k != \"query\"}) or \"\"}")
if code != "200":
    print(f"  HTTP {code}: {body[:300]}"); sys.exit()
r = json.loads(body); m = r["meta"]; qi = m.get("query_interpretation", {})
print(f"  {secs:5.2f}s | status={r[\"status\"]} | intent={qi.get(\"intent\")} ({qi.get(\"parser\")}) | counting={m.get(\"coverage\", {}).get(\"counting_method\")}")
if r["status"] != "ok":
    c = r["clarification"]; print(f"  clarification ({c[\"reason\"]}): {c[\"message\"][:200]}")
    for x in c.get("suggestions", [])[:3]: print("    try:", x)
    sys.exit()
v = r["visualization"]; data = v["data"]; rows = data if isinstance(data, list) else data["nodes"] + data["edges"]
print(f"  type={v[\"type\"]} | datums={len(rows)} | title: {v[\"title\"]}")
for row in rows[:2]:
    print("  row:", {k: (val if k != "nct_ids" else f"[{len(val)} NCT ids]") for k, val in row.items() if k != "citations"})
first = rows[0] if rows else {}
cites = first.get("citations", [])
if cites:
    c = cites[0]; print(f"  cite[0/{first.get(\"citations_total\") or len(cites)}]: {c.get(\"nct_id\", \"(count)\")} {c[\"field_path\"]} = {c[\"excerpt\"][:80]!r}")
val = m.get("validation") or {}
print("  validation:", "PASSED" if val.get("passed") else "FAILED " + str([c["name"] for c in val.get("checks", []) if not c["passed"]]))
notes = [n["code"] for n in m.get("notes", [])]
if notes: print("  notes:", notes)
if r.get("answer_summary"): print("  summary:", r["answer_summary"][:300])
' "$1" "$t0" "$t1"
}

if [ $# -gt 0 ]; then
  run "$("$PY" -c 'import json,sys;print(json.dumps({"query":sys.argv[1]}))' "$1")"; exit
fi

run '{"query":"How has the number of trials for this drug changed over time?","drug_name":"Pembrolizumab"}'
run '{"query":"How many trials started each year for Alzheimer disease since 2015?"}'
run '{"query":"How are breast cancer trials distributed across phases?"}'
run '{"query":"What are the most common intervention types for insomnia trials?"}'
run '{"query":"Compare phases for trials involving semaglutide vs tirzepatide"}'
run '{"query":"Compare sponsor categories across breast cancer and lung cancer trials"}'
run '{"query":"Which countries have the most recruiting trials for type 2 diabetes?"}'
run '{"query":"Show a network of sponsors and drugs for obesity trials since 2020"}'
run '{"query":"Which drugs frequently co-occur in combination studies for melanoma?"}'
run '{"query":"Heatmap of phase by recruitment status across GLP-1 trials","persona":"researcher"}'
run '{"query":"Am I eligible? I am 45, female, BMI 34, taking metformin","nct_id":"NCT03548935"}'
run '{"query":"What side effects did people have?","nct_id":"NCT03548935"}'
run '{"query":"How many patients dropped out and why?","nct_id":"NCT03548935"}'
run '{"query":"Show the survival curve","nct_id":"NCT02578680"}'
run '{"query":"How long will the trial last and how many visits are there?","nct_id":"NCT03548935"}'
run '{"query":"Will the trial cost me anything or reimburse travel?","nct_id":"NCT03548935"}'
# honesty edge cases: no chart, no silent widening
run '{"query":"Which drug is the safest between semaglutide and canagliflozin for diabetes?"}'
run '{"query":"What trials are there for the condition Supercalifragilisticexpialdious since 2020?"}'
run '{"query":"Am I eligible for a semaglutide trial?"}'
