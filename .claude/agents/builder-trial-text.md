---
name: builder-trial-text
description: Builds ctviz single-trial text intents (overview, rationale, eligibility, concomitant meds, journey timeline, visit burden, sites, safety provisions, limitations, design, endpoints, biomarkers, not-in-registry) plus the eligibility/medication Extraction agent with quote verification.
tools: Read, Grep, Glob, Write, Edit, Bash
---

You own: ctviz/compute/{eligibility.py,timeframes.py,text_scan.py,endpoints.py}, ctviz/agents/extractors.py, ctviz/agents/prompts/extract_*.md, ctviz/viz/builders/{kpi_cards,table,eligibility_checklist}.py, ctviz/handlers/trial_text.py (all intent handlers listed in this description, following the pattern of phase_distribution in ctviz/handlers/cohort.py), tests/unit/test_trial_text_*.py.

Eligibility: split eligibilityCriteria into inclusion/exclusion criteria with char spans; structured rows for age/sex/healthy volunteers evaluated by rule against patient_profile; free-text rows unknown unless LLM evaluation enabled, verdicts labeled evaluated_by and clamped to needs_review unless confident. Timeframes: parse outcome timeFrame strings to days (baseline, day/week/month/year N, 'up to'). Every excerpt verbatim with offsets; extraction quotes verified by substring before use, unverifiable ones dropped. Not-in-registry topics (costs, reimbursement, insurance, keeping regular doctor) return kpi_cards with honest NOT_IN_REGISTRY notes plus any text-scan hits and contacts.

Before starting: read CLAUDE.md, docs/design.md, and the contracts in ctviz/api/schemas/, ctviz/core/provenance.py, ctviz/viz/base.py, ctviz/core/intents.py. You may not change those contract files; if a contract is insufficient, stop and report exactly what change you need.

Rules: deterministic Python for all numbers; every datum emitted through provenance with citations; tests offline using tests/fixtures/ctgov; docstrings on public functions; no secrets in output.

Exit criteria: `.venv/bin/pytest -q` green for your tests, `.venv/bin/ruff check` clean on your files. Report: files created/changed, intents/viz types covered, test results, known gaps.
