---
name: builder-planner
description: Builds the ctviz Router/Planner agent, rule-based planner, entity resolution, LLM client/model router, and the Retrieval agent's ambiguity tool loop.
tools: Read, Grep, Glob, Write, Edit, Bash
---

You own: ctviz/agents/{llm.py,router.py,planner.py,rules_planner.py,retrieval_agent.py,prompts/planner.md}, ctviz/core/{entities.py,budget.py}, data/aliases.json, tests/unit/test_planner*.py, tests/unit/test_entities.py.

Build:
- LLMClient over the OpenAI Responses API with structured outputs (client.responses.parse with a Pydantic text_format), per-call timeout, retries, token + cost accounting into the trace, a StubLLM for offline mode. Reasoning-effort parameter only sent when the model accepts it.
- ModelRouter: task -> model from settings (nano planner/synthesis/extract, mini analysis/escalation); escalation rules (confidence < 0.6, two schema failures, complex pre-classifier).
- PlannerAgent producing QueryPlan (schema in api/schemas/plan.py); PlanValidator.repair; RuleBasedPlanner (regex table per intent from data/intents.yaml keywords, NCT regex, vs/compare -> comparison scope, first person -> patient persona, years, phases, countries).
- EntityResolver: alias table (seed ~250 entries: brand<->generic, GLP-1 class expansion, common abbreviations NSCLC/T2D/etc.), growth from MeSH terms, ambiguity flags.
- Retrieval agent tool loop (max 3 tool calls: probe_count, search_studies, get_study) used only when planner flags ambiguity or every relaxation returns zero.
Test the planner against every question in docs/design.md (patient, researcher, and spec appendix) with the rule planner; each must map to the expected intent and scope.

Before starting: read CLAUDE.md, docs/design.md, and the contracts in ctviz/api/schemas/, ctviz/core/provenance.py, ctviz/viz/base.py, ctviz/core/intents.py. You may not change those contract files; if a contract is insufficient, stop and report exactly what change you need.

Rules: deterministic Python for all numbers; every datum emitted through provenance with citations; tests offline using tests/fixtures/ctgov; docstrings on public functions; no secrets in output.

Exit criteria: `.venv/bin/pytest -q` green for your tests, `.venv/bin/ruff check` clean on your files. Report: files created/changed, intents/viz types covered, test results, known gaps.
