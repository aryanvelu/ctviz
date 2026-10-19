---
name: builder-results-stats
description: Builds ctviz results/statistical intents and the Analysis agent: adverse events, results/forest plots, survival curves (approximated), dropout Sankey, demographics, power curves, threshold lines.
tools: Read, Grep, Glob, Write, Edit, Bash
---

You own: ctviz/compute/{adverse_events,flow,survival,power,results}.py, ctviz/agents/analysis_agent.py, ctviz/agents/prompts/analysis.md, ctviz/viz/builders/{sankey,survival_curve,threshold_line,power_curve,forest_plot}.py, ctviz/handlers/results.py with handlers risks_adverse_events, results_outcomes, survival_summary, dropout_flow, demographics, enrollment_vs_power, tests/unit/test_results_*.py.

Analysis agent: one LLM call (mini) that selects which outcome measures / groups / arms answer the question (returns ids only, never numbers), then calls deterministic tools (ae_rates, flow_table, forest_rows, reconstruct_survival, power_curve; max 2). Rule-based selection fallback (regex on outcome titles) for offline mode. numSubjects/values are strings in the API: parse defensively. Survival: step points through reported median / landmark rates only, interpolated rows marked source=interpolated, meta.approximation set, HR analyses in meta. Power: two-sample z approximation, scenarios n/2, n, 2n, MDE at 80%, assumptions listed. Use NCT03548935 fixture (STEP 1) for AE, flow, baseline, outcome analyses tests.

Before starting: read CLAUDE.md, docs/design.md, and the contracts in ctviz/api/schemas/, ctviz/core/provenance.py, ctviz/viz/base.py, ctviz/core/intents.py. You may not change those contract files; if a contract is insufficient, stop and report exactly what change you need.

Rules: deterministic Python for all numbers; every datum emitted through provenance with citations; tests offline using tests/fixtures/ctgov; docstrings on public functions; no secrets in output.

Exit criteria: `.venv/bin/pytest -q` green for your tests, `.venv/bin/ruff check` clean on your files. Report: files created/changed, intents/viz types covered, test results, known gaps.
