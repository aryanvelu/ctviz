"""Command line: ``ctviz serve | ask | examples | bench | warm | record``."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import sys
import time
from pathlib import Path
from typing import Any

from ctviz.config import PROJECT_ROOT, Settings, get_settings

EXAMPLES_DIR = PROJECT_ROOT / "examples"
EXAMPLES_FILE = EXAMPLES_DIR / "requests.json"
WARM_FILE = PROJECT_ROOT / "data" / "warm_queries.yaml"
DEFAULT_BENCH = [
    (
        "pembrolizumab_trend",
        {
            "query": "How has the number of pembrolizumab trials changed per year since 2015?",
            "drug_name": "Pembrolizumab",
            "start_year": 2015,
        },
    ),
    (
        "semaglutide_vs_tirzepatide_phases",
        {"query": "Compare phases for semaglutide vs tirzepatide", "drug_names": ["semaglutide", "tirzepatide"]},
    ),
    ("step1_overview", {"query": "What is this trial about?", "nct_id": "NCT03548935"}),
]


# -- helpers --------------------------------------------------------------------------------------
def _load_named_requests(path: Path) -> list[tuple[str, dict[str, Any]]]:
    """Accepts ``[{"name", "request"}]``, ``{"queries": [{"id", "body"}]}`` (data/warm_queries.yaml),
    ``{name: request}`` or a plain list of requests (JSON or YAML)."""
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(text) or []
    else:
        data = json.loads(text)
    if isinstance(data, dict) and isinstance(data.get("queries"), list):
        data = data["queries"]
    if isinstance(data, dict):
        return [(str(k), v) for k, v in data.items()]
    out = []
    for i, item in enumerate(data):
        if isinstance(item, dict) and "request" in item:
            out.append((str(item.get("name") or f"q{i + 1}"), item["request"]))
        elif isinstance(item, dict) and "body" in item:
            out.append((str(item.get("id") or item.get("name") or f"q{i + 1}"), item["body"]))
        else:
            out.append((f"q{i + 1}", item))
    return out


class _Services:
    """Builds and tears down the same services the API uses (in-process, no server)."""

    def __init__(self, settings: Settings, llm: Any = None, **orch_kwargs: Any):
        import ctviz.handlers  # noqa: F401
        from ctviz.app import build_services

        self.s = build_services(settings)
        if llm is not None or orch_kwargs:
            from ctviz.core.orchestrator import Orchestrator

            self.s["llm"] = llm or self.s["llm"]
            self.s["orchestrator"] = Orchestrator(
                settings,
                self.s["db"],
                self.s["client"],
                self.s["llm"],
                self.s["graph"],
                text_index=self.s["text_index"],
                **orch_kwargs,
            )
        self.orch = self.s["orchestrator"]

    async def close(self) -> None:
        from ctviz.core.background import drain

        await drain(timeout_s=15.0)
        await self.s["client"].aclose()
        self.s["db"].close()


def _settings(offline: bool = False, **overrides: Any) -> Settings:
    if offline:
        os.environ["CTVIZ_OFFLINE"] = "1"
    settings = get_settings()
    return settings.model_copy(update=overrides) if overrides else settings


def _dump(resp: Any, debug: bool = False, citation_detail: str = "sample") -> str:
    from ctviz.api.public import to_public

    return json.dumps(to_public(resp, debug=debug, citation_detail=citation_detail), indent=2)


# -- commands -------------------------------------------------------------------------------------
def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("ctviz.app:create_app", factory=True, host=args.host, port=args.port, reload=args.reload)
    return 0


def _parse_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def build_ask_request(args: argparse.Namespace) -> dict[str, Any]:
    """Turn ``ctviz ask`` flags into a request body (``--field key=value`` accepts JSON values)."""
    body: dict[str, Any] = {"query": args.question}
    if args.nct_id:
        if len(args.nct_id) == 1:
            body["nct_id"] = args.nct_id[0]
        else:
            body["nct_ids"] = args.nct_id
    if args.drug:
        body.update({"drug_name": args.drug[0]} if len(args.drug) == 1 else {"drug_names": args.drug})
    if args.condition:
        body.update({"condition": args.condition[0]} if len(args.condition) == 1 else {"conditions": args.condition})
    for item in args.field or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--field expects key=value, got {item!r}")
        body[key.strip()] = _parse_value(value)
    if getattr(args, "citations", None):
        body["citation_detail"] = args.citations
    return body


def _explain(resp: Any, rec: Any) -> None:
    out = sys.stdout
    plan = (rec.artifacts.get("plan") or {}).get("plan", {})
    out.write("== Plan ==\n")
    for key in (
        "intent",
        "intent_confidence",
        "persona",
        "scope",
        "parser",
        "nct_ids",
        "group_by",
        "series_by",
        "preferred_visualization",
        "clarification",
    ):
        if plan.get(key) not in (None, [], ""):
            out.write(f"  {key}: {plan[key]}\n")
    for c in plan.get("cohorts", []):
        out.write(f"  cohort: {json.dumps({k: v for k, v in c.items() if v not in (None, [], False)})}\n")
    for a in plan.get("assumptions", []):
        out.write(f"  assumption: {a}\n")
    out.write(f"  filters_applied: {json.dumps(resp.meta.filters_applied)}\n")
    out.write("== Steps ==\n")
    for s in rec.steps:
        out.write(f"  {s['tool']:<32} {s['ms']:>8.1f} ms  {s['status']:<8} {json.dumps(s.get('summary'))[:200]}\n")
    out.write("== LLM calls ==\n")
    trace = resp.meta.trace
    calls = trace.llm_calls if trace is not None else []
    if not calls:
        out.write("  (none: answered by rules / no LLM needed)\n")
    for c in calls:
        out.write(
            f"  {c.agent:<20} {c.model:<14} {c.ms:>8.1f} ms  in={c.input_tokens} (cached {c.cached_tokens}) "
            f"out={c.output_tokens}  ${c.cost_usd:.6f}\n"
        )
    if calls:
        out.write("  (the LLM's raw proposal and every repair are in plan.json; the validated plan is shown above)\n")
    out.write("== ClinicalTrials.gov calls ==\n")
    for c in rec.ctgov_calls:
        out.write(
            f"  [{'cached' if c.get('cached') else 'live':<6}] {c.get('ms', 0):>7.1f} ms  "
            f"{c.get('status')}  {c.get('url')}\n"
        )
        if c.get("cache_file"):
            out.write(f"           raw: {c['cache_file']}\n")
    facts = (rec.artifacts.get("handler_result") or {}).get("facts") or {}
    if facts:
        out.write("== Counts (handler facts) ==\n")
        for k, v in list(facts.items())[:60]:
            out.write(f"  {k}: {v}\n")
    out.write("== Notes ==\n")
    for n in resp.meta.notes:
        out.write(f"  {n.code.value} ({n.severity}): {n.message}\n")
    if rec.directory is not None:
        out.write(f"== Run artifacts: {rec.directory} ==\n")
    out.write("== Response ==\n")


def cmd_ask(args: argparse.Namespace) -> int:
    from ctviz.api.schemas.request import VisualizeRequest

    body = build_ask_request(args)
    body.setdefault("include_trace", True)  # --explain reads the trace; the printed JSON adds it only with --debug
    request = VisualizeRequest.model_validate(body)

    async def go() -> int:
        svc = _Services(_settings(offline=args.offline, runs_enabled=True))  # `ask` always keeps run artifacts
        try:
            resp, rec = await svc.orch.run_recorded(request)
        finally:
            await svc.close()
        if args.explain:
            _explain(resp, rec)
        text = _dump(resp, debug=args.debug, citation_detail=request.citation_detail)
        if args.out:
            pathlib.Path(args.out).write_text(text + "\n")
            print(f"public JSON written to {args.out}", file=sys.stderr)
        if not args.quiet:
            print(text)
        return 0

    return asyncio.run(go())


def cmd_examples(args: argparse.Namespace) -> int:
    from ctviz.api.schemas.request import VisualizeRequest

    if not EXAMPLES_FILE.exists():
        print(f"{EXAMPLES_FILE} not found; nothing to run", file=sys.stderr)
        return 1
    named = _load_named_requests(EXAMPLES_FILE)

    async def go() -> None:
        svc = _Services(_settings(offline=args.offline))
        try:
            for name, body in named:
                t0 = time.perf_counter()
                resp = await svc.orch.run(VisualizeRequest.model_validate(body))
                from ctviz.api.public import to_public

                payload = {"name": name, "request": body, "response": to_public(resp)}
                (EXAMPLES_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")
                kind = resp.visualization.type.value if resp.visualization else resp.status
                print(f"{name:<40} {kind:<22} {(time.perf_counter() - t0) * 1000:>8.0f} ms")
        finally:
            await svc.close()

    asyncio.run(go())
    return 0


def _pct(values: list[float], q: float) -> str:
    from ctviz.obs.metrics import percentile

    v = percentile(values, q)
    return "-" if v is None else f"{v:,.0f}"


def cmd_bench(args: argparse.Namespace) -> int:
    from ctviz.api.schemas.request import VisualizeRequest

    named = _load_named_requests(EXAMPLES_FILE) if EXAMPLES_FILE.exists() else DEFAULT_BENCH
    if args.limit:
        named = named[: args.limit]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    async def go() -> list[tuple[str, dict[str, list[float]]]]:
        settings = _settings(offline=args.offline)
        svc = _Services(settings)
        from ctviz.core.orchestrator import Orchestrator

        # same services, response cache disabled: measures the pipeline over a warm HTTP cache
        warm_orch = Orchestrator(
            settings,
            svc.s["db"],
            svc.s["client"],
            svc.s["llm"],
            svc.s["graph"],
            text_index=svc.s["text_index"],
            use_response_cache=False,
            runs_dir=None,
        )
        rows = []
        try:
            for name, body in named:
                times: dict[str, list[float]] = {m: [] for m in modes}
                for _ in range(args.runs):
                    for mode in modes:
                        req = VisualizeRequest.model_validate({**body, "refresh": mode == "cold"})
                        orch = warm_orch if mode == "warm" else svc.orch
                        t0 = time.perf_counter()
                        await orch.run(req)
                        times[mode].append((time.perf_counter() - t0) * 1000)
                rows.append((name, times))
        finally:
            await svc.close()
        return rows

    rows = asyncio.run(go())
    header = f"{'query':<40}" + "".join(f"{m + ' p50':>12}{m + ' p95':>12}" for m in modes)
    print(header)
    print("-" * len(header))
    for name, times in rows:
        print(f"{name[:40]:<40}" + "".join(f"{_pct(times[m], 50):>12}{_pct(times[m], 95):>12}" for m in modes))
    print(f"(ms; {args.runs} run(s) per mode; cold = refresh=true, warm = HTTP cache only, cached = response cache)")
    return 0


def cmd_warm(args: argparse.Namespace) -> int:
    from ctviz.api.schemas.request import VisualizeRequest

    if not WARM_FILE.exists():
        print(f"{WARM_FILE} not found; nothing to warm", file=sys.stderr)
        return 0
    named = _load_named_requests(WARM_FILE)

    async def go() -> None:
        svc = _Services(_settings(offline=args.offline))
        try:
            for item in [] if args.queries_only else _snapshot_list(WARM_FILE):
                t0 = time.perf_counter()
                n = await _warm_snapshot(svc, item)
                print(
                    f"snapshot {item.get('condition') or item.get('drug'):<38} {n:>7,} trials "
                    f"{(time.perf_counter() - t0):>6.0f} s"
                )
            for name, body in [] if args.snapshots_only else named:
                t0 = time.perf_counter()
                await svc.orch.run(VisualizeRequest.model_validate(body))
                print(f"warmed {name:<40} {(time.perf_counter() - t0) * 1000:>8.0f} ms")
        finally:
            await svc.close()

    asyncio.run(go())
    return 0


def _snapshot_list(path: Path) -> list[dict[str, Any]]:
    import yaml

    data = yaml.safe_load(path.read_text()) or {}
    return [s for s in data.get("snapshots") or [] if isinstance(s, dict)] if isinstance(data, dict) else []


async def _warm_snapshot(svc: _Services, item: dict[str, Any]) -> int:
    """Read every record of one cohort (up to ``max``) so later questions use its snapshot."""
    from ctviz.api.schemas.plan import CohortFilter
    from ctviz.core.background import drain
    from ctviz.ctgov.fields import FieldSet
    from ctviz.ctgov.retrieval import fetch_cohort

    name = str(item.get("condition") or item.get("drug"))
    cf = CohortFilter(
        label=name, conditions=[name] if item.get("condition") else [], drugs=[name] if item.get("drug") else []
    )
    data = await fetch_cohort(
        svc.s["client"], cf, field_set=FieldSet.LANDSCAPE, max_studies=int(item.get("max", 60000))
    )
    await drain(timeout_s=120.0)  # study records + snapshot are written in the background
    return len(data.studies)


def cmd_record(args: argparse.Namespace) -> int:
    """Capture fixtures for offline replay. Uses the stub LLM so plans match offline (rule) plans exactly."""
    from ctviz.agents.llm import StubLLM
    from ctviz.api.schemas.request import VisualizeRequest

    named = _load_named_requests(EXAMPLES_FILE) if EXAMPLES_FILE.exists() else DEFAULT_BENCH
    settings = _settings(record=True, offline=False)

    async def go() -> None:
        svc = _Services(settings, llm=StubLLM(), use_response_cache=False)
        try:
            await svc.s["client"].version()
            for name, body in named:
                resp = await svc.orch.run(VisualizeRequest.model_validate({**body, "refresh": True}))
                print(f"recorded {name:<40} {resp.visualization.type.value if resp.visualization else resp.status}")
        finally:
            await svc.close()

    asyncio.run(go())
    print(f"fixtures written to {settings.fixtures_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ctviz", description="ClinicalTrials.gov query-to-visualization agent")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(fn=cmd_serve)

    a = sub.add_parser("ask", help="answer one question in-process")
    a.add_argument("question")
    a.add_argument("--nct-id", action="append")
    a.add_argument("--drug", action="append")
    a.add_argument("--condition", action="append")
    a.add_argument("--field", action="append", metavar="KEY=VALUE", help="any request field (JSON values ok)")
    a.add_argument("--explain", action="store_true", help="print plan, steps, CT.gov calls, facts and notes")
    a.add_argument("--debug", action="store_true", help="include meta.debug (trace, cache, API params) in the JSON")
    a.add_argument("--out", help="also write the public JSON (what a client receives) to this file")
    a.add_argument("--quiet", action="store_true", help="do not print the JSON (use with --out)")
    a.add_argument("--citations", choices=["ids_only", "sample", "full"], help="citation_detail (default sample)")
    a.add_argument("--offline", action="store_true")
    a.set_defaults(fn=cmd_ask)

    w = sub.add_parser("warm", help="build cohort snapshots and run data/warm_queries.yaml to fill caches")
    w.add_argument("--offline", action="store_true")
    w.add_argument("--snapshots-only", action="store_true", help="only build the cohort snapshots")
    w.add_argument("--queries-only", action="store_true", help="only run the warm queries")
    w.set_defaults(fn=cmd_warm)
    e = sub.add_parser("examples", help="regenerate examples/<name>.json")
    e.add_argument("--offline", action="store_true")
    e.set_defaults(fn=cmd_examples)

    b = sub.add_parser("bench", help="latency p50/p95 per query")
    b.add_argument("--runs", type=int, default=3)
    b.add_argument("--modes", default="cold,warm,cached", help="comma list of cold,warm,cached")
    b.add_argument("--limit", type=int, default=0, help="only the first N queries")
    b.add_argument("--offline", action="store_true")
    b.set_defaults(fn=cmd_bench)

    r = sub.add_parser("record", help="capture CT.gov fixtures for offline replay (network)")
    r.set_defaults(fn=cmd_record)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
