"""Extra checks for the results handlers: independent recomputation from raw JSON, survival anchors,
power reference values, deep citation invariant, LLM rationale isolation, and group-id matching.

Expected values here are recomputed directly from the recorded JSON (not via ``ctviz.compute``),
so a bug shared by the module and its own tests would still be caught. Offline, mocked LLM only.
"""

from __future__ import annotations

import json
import math

import pytest

from ctviz.agents.analysis_agent import LLMPick
from ctviz.agents.llm import StubLLM
from ctviz.api.schemas import Intent
from ctviz.api.schemas.viz import GraphData, VizType
from ctviz.compute import power as pw
from ctviz.core.provenance import Study, verify_citation
from ctviz.ctgov.paths import path_get
from ctviz.handlers import results as R
from tests.conftest import load_study_raw

ALL = ["NCT03548935", "NCT02578680", "NCT02142738", "NCT01844505", "NCT01284335", "NCT07654361"]
HANDLERS = [
    (R.risks_adverse_events, Intent.RISKS_ADVERSE_EVENTS),
    (R.results_outcomes, Intent.RESULTS_OUTCOMES),
    (R.survival_summary, Intent.SURVIVAL_SUMMARY),
    (R.dropout_flow, Intent.DROPOUT_FLOW),
    (R.demographics, Intent.DEMOGRAPHICS),
    (R.enrollment_vs_power, Intent.ENROLLMENT_VS_POWER),
]
HANDLER_IDS = [h[1].value for h in HANDLERS]


def _rows(viz):
    return [d.model_extra or {} for d in viz.data]


def _items(viz):
    return viz.data.nodes + viz.data.edges if isinstance(viz.data, GraphData) else viz.data


async def _run(handler, make_ctx, s: Study, intent, query="test", llm=None):
    ctx = make_ctx(intent, trials=[s], query=query)
    ctx.llm = llm
    return await handler(ctx)


# == (1) independent recomputation on STEP 1 =======================================================

STEP1 = "NCT03548935"


async def test_step1_ae_rates_match_raw(make_ctx):
    raw = load_study_raw(STEP1)
    aem = raw["resultsSection"]["adverseEventsModule"]
    titles = {g["id"]: g["title"] for g in aem["eventGroups"]}
    # five (term, kind) pairs recomputed straight from the JSON
    expected: dict[tuple[str, str], float] = {}
    for g in aem["eventGroups"]:
        expected[("Any serious adverse event", titles[g["id"]])] = g["seriousNumAffected"] / g["seriousNumAtRisk"]
    for term in ("Nausea", "Diarrhoea", "Vomiting"):
        ev = next(e for e in aem["otherEvents"] if e["term"] == term)
        st = next(x for x in ev["stats"] if x["groupId"] == aem["eventGroups"][0]["id"])
        expected[(term, titles[st["groupId"]])] = int(st["numAffected"]) / int(st["numAtRisk"])
    assert len(expected) == 5

    res = await _run(R.risks_adverse_events, make_ctx, Study(raw), Intent.RISKS_ADVERSE_EVENTS)
    got = {(r["term"], r["arm"]): r for r in _rows(res.visualization) if r["seriousness"] in ("other", "any_serious")}
    for key, rate in expected.items():
        assert key in got, f"missing AE row {key}"
        assert got[key]["rate"] == pytest.approx(rate, abs=1e-4)
        assert got[key]["n_affected"] / got[key]["n_at_risk"] == pytest.approx(rate, abs=1e-12)


async def test_step1_sankey_completed_matches_raw(make_ctx):
    raw = load_study_raw(STEP1)
    pf = raw["resultsSection"]["participantFlowModule"]
    titles = {g["id"]: g["title"] for g in pf["groups"]}
    ms = {m["type"]: m for m in pf["periods"][0]["milestones"]}
    started = {titles[a["groupId"]]: int(a["numSubjects"]) for a in ms["STARTED"]["achievements"]}
    completed = {titles[a["groupId"]]: int(a["numSubjects"]) for a in ms["COMPLETED"]["achievements"]}
    not_completed = {titles[a["groupId"]]: int(a["numSubjects"]) for a in ms["NOT COMPLETED"]["achievements"]}

    res = await _run(R.dropout_flow, make_ctx, Study(raw), Intent.DROPOUT_FLOW)
    g = res.visualization.data
    edges = [e.model_extra for e in g.edges]
    starts = {n.model_extra["group"]: n.model_extra["value"] for n in g.nodes if n.id.endswith(":start")}
    assert starts == started
    for arm in started:
        arm_edges = [e for e in edges if e["group"] == arm]
        done = sum(e["weight"] for e in arm_edges if e["type"] == "completed")
        not_done = sum(e["weight"] for e in arm_edges if e["type"] != "completed")
        assert done == completed[arm]
        assert not_done == not_completed[arm] == started[arm] - completed[arm]


async def test_step1_baseline_sex_counts_match_raw(make_ctx):
    raw = load_study_raw(STEP1)
    bm = raw["resultsSection"]["baselineCharacteristicsModule"]
    titles = {g["id"]: g["title"] for g in bm["groups"]}
    sex = next(m for m in bm["measures"] if m["title"].lower().startswith("sex"))
    expected = {}
    for cat in sex["classes"][0]["categories"]:
        for meas in cat["measurements"]:
            if titles[meas["groupId"]] != "Total":
                expected[(titles[meas["groupId"]], cat["title"])] = int(meas["value"])
    assert len(expected) == 4

    res = await _run(R.demographics, make_ctx, Study(raw), Intent.DEMOGRAPHICS, query="sex breakdown")
    got = {(r["group"], r["category"]): r["n"] for r in _rows(res.visualization) if r["characteristic"] == "Sex"}
    assert got == expected


# == (2) KEYNOTE-189 survival: reported points come from the JSON ==================================


def _measure_values(raw: dict) -> tuple[set[float], set[float]]:
    """(median values, all numeric values) over every outcome-measure measurement in the record."""
    medians: set[float] = set()
    values: set[float] = set()
    for om in raw["resultsSection"]["outcomeMeasuresModule"]["outcomeMeasures"]:
        for cls in om.get("classes", []):
            for cat in cls.get("categories", []):
                for m in cat.get("measurements", []):
                    try:
                        v = float(m.get("value"))
                    except (TypeError, ValueError):
                        continue
                    values.add(v)
                    if om.get("paramType") == "MEDIAN":
                        assert "month" in om.get("unitOfMeasure", "").lower()  # times compared in months
                        medians.add(v)
    return medians, values


@pytest.mark.parametrize("nct", ["NCT02578680", "NCT01844505"])
@pytest.mark.parametrize("query", ["Show the survival curve", "overall survival", "progression-free survival"])
async def test_survival_reported_points_exist_in_raw(make_ctx, nct, query):
    raw = load_study_raw(nct)
    medians, values = _measure_values(raw)
    res = await _run(R.survival_summary, make_ctx, Study(raw), Intent.SURVIVAL_SUMMARY, query=query)
    viz = res.visualization
    assert viz.type is VizType.SURVIVAL_CURVE
    rows = _rows(viz)
    reported = [d for d in viz.data if (d.model_extra or {})["source"] == "reported"]
    assert reported
    for d in reported:
        r = d.model_extra
        if r["time"] == 0:
            assert r["survival"] == 1.0
            continue
        if r["survival"] == 0.5 and r["time"] in medians:
            src = path_get(raw, d.citations[0].field_path)
            assert float(src) == r["time"]
        else:
            # landmark rate: the registry gives a percent (KEYNOTE-189 style) or, despite a "Percentage"
            # unit, a fraction (CheckMate 067 style); the point must equal the cited leaf either way
            def same(v: float, surv: float = r["survival"]) -> bool:
                return math.isclose(surv * 100, v, abs_tol=1e-6) or (v <= 1 and math.isclose(surv, v, abs_tol=1e-9))

            assert any(same(v) for v in values), r
            cited = [float(path_get(raw, c.field_path)) for c in d.citations if c.field_path.endswith(".value")]
            assert cited and same(cited[0]), (r, cited)
    for series in {r["series"] for r in rows}:
        last = max(r["time"] for r in rows if r["series"] == series and r["source"] == "reported")
        assert all(r["time"] <= last + 1e-9 for r in rows if r["series"] == series), series
        curve = sorted((r["time"], r["survival"]) for r in rows if r["series"] == series)
        assert all(b[1] <= a[1] + 1e-9 for a, b in zip(curve, curve[1:], strict=False)), series


# == (3) power against textbook references =========================================================


@pytest.mark.parametrize("n,d,ref", [(64, 0.5, 0.80), (100, 0.3, 0.56)])
def test_power_matches_reference(n, d, ref):
    assert pw.power(d, n) == pytest.approx(ref, abs=0.01)


def _two_arm_study(n_per_arm: int) -> Study:
    gs = [{"id": "FG000", "title": "Drug"}, {"id": "FG001", "title": "Placebo"}]
    ach = [{"groupId": g["id"], "numSubjects": str(n_per_arm)} for g in gs]
    return Study(
        {
            "protocolSection": {"identificationModule": {"nctId": "NCT00000064"}},
            "hasResults": True,
            "resultsSection": {
                "participantFlowModule": {
                    "groups": gs,
                    "periods": [
                        {
                            "title": "Overall Study",
                            "milestones": [
                                {"type": "STARTED", "achievements": ach},
                                {"type": "COMPLETED", "achievements": ach},
                            ],
                        }
                    ],
                }
            },
        }
    )


@pytest.mark.parametrize("n,d,ref", [(64, 0.5, 0.80), (100, 0.3, 0.56)])
async def test_power_handler_curve_matches_reference(make_ctx, n, d, ref):
    res = await _run(R.enrollment_vs_power, make_ctx, _two_arm_study(n), Intent.ENROLLMENT_VS_POWER)
    rows = _rows(res.visualization)
    pt = next(r for r in rows if r["n_per_arm"] == n and r["effect_size"] == pytest.approx(d))
    assert pt["power"] == pytest.approx(ref, abs=0.01)


# == (5)/(6) synthetic study with shuffled group orders ============================================

ARMS = ["Alpha", "Bravo", "Charlie", "Delta", "Echo"]


def _vals(k: int) -> dict[str, float]:
    started = 100 + 10 * k
    return {
        "started": started,
        "completed": started - (3 + k),
        "female": 40 + k,
        "male": started - (40 + k),
        "weight": -(k + 1.5),
        "serious": 5 + k,
        "nausea": 20 + 2 * k,
    }


def synthetic_study(n_arms: int = 3) -> Study:
    """Every module lists its groups (and its per-group entries) in a different order with a different id prefix."""
    arms = ARMS[:n_arms]
    idx = list(range(n_arms))
    rev = idx[::-1]
    rot = idx[1:] + idx[:1]
    rot2 = idx[2:] + idx[:2]

    def gid(prefix, k):
        return f"{prefix}{(k * 7 + 3) % 10}{k}"  # ids unrelated to list position

    flow_groups = [{"id": gid("FG", k), "title": arms[k]} for k in rev]

    def ach(order, key):
        return [{"groupId": gid("FG", k), "numSubjects": str(int(_vals(k)[key]))} for k in order]

    flow = {
        "groups": flow_groups,
        "periods": [
            {
                "title": "Overall Study",
                "milestones": [
                    {"type": "STARTED", "achievements": ach(rot, "started")},
                    {"type": "COMPLETED", "achievements": ach(rot2, "completed")},
                    {
                        "type": "NOT COMPLETED",
                        "achievements": [
                            {
                                "groupId": gid("FG", k),
                                "numSubjects": str(int(_vals(k)["started"] - _vals(k)["completed"])),
                            }
                            for k in idx
                        ],
                    },
                ],
                "dropWithdraws": [
                    {
                        "type": "Adverse Event",
                        "reasons": [{"groupId": gid("FG", k), "numSubjects": str(3 + k)} for k in rev],
                    }
                ],
            }
        ],
    }
    base_groups = [{"id": gid("BG", k), "title": arms[k]} for k in rot] + [{"id": "BG999", "title": "Total"}]
    baseline = {
        "groups": base_groups,
        "measures": [
            {
                "title": "Sex: Female, Male",
                "paramType": "COUNT_OF_PARTICIPANTS",
                "classes": [
                    {
                        "categories": [
                            {
                                "title": "Female",
                                "measurements": [
                                    {"groupId": gid("BG", k), "value": str(_vals(k)["female"])} for k in rev
                                ]
                                + [{"groupId": "BG999", "value": str(sum(_vals(k)["female"] for k in idx))}],
                            },
                            {
                                "title": "Male",
                                "measurements": [
                                    {"groupId": gid("BG", k), "value": str(int(_vals(k)["male"]))} for k in rot2
                                ],
                            },
                        ]
                    }
                ],
            }
        ],
    }
    outcomes = {
        "outcomeMeasures": [
            {
                "type": "PRIMARY",
                "title": "Change in body weight",
                "paramType": "MEAN",
                "dispersionType": "Standard Deviation",
                "unitOfMeasure": "kg",
                "timeFrame": "Week 52",
                "groups": [{"id": gid("OG", k), "title": arms[k]} for k in rot2],
                "classes": [
                    {
                        "categories": [
                            {
                                "measurements": [
                                    {"groupId": gid("OG", k), "value": str(_vals(k)["weight"]), "spread": "1.0"}
                                    for k in rev
                                ]
                            }
                        ]
                    }
                ],
            }
        ]
    }
    ae = {
        "frequencyThreshold": "5",
        "eventGroups": [
            {
                "id": gid("EG", k),
                "title": arms[k],
                "seriousNumAffected": _vals(k)["serious"],
                "seriousNumAtRisk": _vals(k)["started"],
            }
            for k in rot
        ],
        "otherEvents": [
            {
                "term": "Nausea",
                "organSystem": "Gastrointestinal disorders",
                "stats": [
                    {"groupId": gid("EG", k), "numAffected": _vals(k)["nausea"], "numAtRisk": _vals(k)["started"]}
                    for k in rev
                ],
            }
        ],
    }
    return Study(
        {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT09999999"},
                "designModule": {"phases": ["PHASE3"], "enrollmentInfo": {"count": 999, "type": "ACTUAL"}},
            },
            "hasResults": True,
            "resultsSection": {
                "participantFlowModule": flow,
                "baselineCharacteristicsModule": baseline,
                "outcomeMeasuresModule": outcomes,
                "adverseEventsModule": ae,
            },
        }
    )


async def test_group_ids_pair_numbers_with_right_arm(make_ctx):
    s = synthetic_study(3)
    want = {ARMS[k]: _vals(k) for k in range(3)}

    flow = (await _run(R.dropout_flow, make_ctx, s, Intent.DROPOUT_FLOW)).visualization.data
    starts = {n.model_extra["group"]: n.model_extra["value"] for n in flow.nodes if n.id.endswith(":start")}
    assert starts == {a: v["started"] for a, v in want.items()}
    done = {e.model_extra["group"]: e.model_extra["weight"] for e in flow.edges if e.model_extra["type"] == "completed"}
    assert done == {a: v["completed"] for a, v in want.items()}
    for e in flow.edges:
        if e.model_extra["type"] == "dropout":
            k = ARMS.index(e.model_extra["group"])
            assert e.model_extra["weight"] == 3 + k

    demo = _rows((await _run(R.demographics, make_ctx, s, Intent.DEMOGRAPHICS, query="sex")).visualization)
    got = {(r["group"], r["category"]): r["n"] for r in demo}
    assert got == {
        **{(a, "Female"): v["female"] for a, v in want.items()},
        **{(a, "Male"): v["male"] for a, v in want.items()},
    }

    out = _rows(
        (await _run(R.results_outcomes, make_ctx, s, Intent.RESULTS_OUTCOMES, query="body weight")).visualization
    )
    assert {r["group"]: r["value"] for r in out} == {a: v["weight"] for a, v in want.items()}

    ae_rows = _rows((await _run(R.risks_adverse_events, make_ctx, s, Intent.RISKS_ADVERSE_EVENTS)).visualization)
    assert {(r["arm"], r["seriousness"]) for r in ae_rows} == {(a, k) for a in want for k in ("any_serious", "other")}
    for r in ae_rows:
        v = want[r["arm"]]
        assert r["n_at_risk"] == v["started"]
        key = "serious" if r["seriousness"] == "any_serious" else "nausea"
        assert r["n_affected"] == v[key], r

    power = (await _run(R.enrollment_vs_power, make_ctx, s, Intent.ENROLLMENT_VS_POWER)).visualization
    assert power.meta.stats["power_inputs"]["n_per_arm"] == pytest.approx(
        round(pw.harmonic_mean([v["started"] for v in want.values()]), 1)
    )


# == (4) deep citation invariant ===================================================================


def _check_all_citations(res, raw: dict) -> None:
    viz = res.visualization
    noted = {i for n in res.notes for i in n.affects}
    for d in _items(viz):
        for c in d.citations:
            assert verify_citation(c, raw), f"{d.id}: unverified {c.field_path} = {c.excerpt!r}"
        if not d.citations:
            assert (d.model_extra or {}).get("cite_note"), f"{d.id}: uncited datum without cite_note"
            assert d.id in noted or "*" in noted or len(_items(viz)) == 1, d.id
    for did, refs in res.full_refs.items():  # the full evidence list, not just the first N shown
        for r in refs:
            assert verify_citation(r.to_citation(), raw), f"{did}: unverified full ref {r.field_path}"
    for t in viz.meta.thresholds:
        for c in t.citations:
            assert str(path_get(raw, c.field_path)) == c.excerpt
    for a in viz.meta.stats.get("age_summary", []) or []:
        for c in a["citations"]:
            assert str(path_get(raw, c["field_path"])) == c["excerpt"]


@pytest.mark.parametrize("nct", ALL + ["synthetic"])
@pytest.mark.parametrize("handler,intent", HANDLERS, ids=HANDLER_IDS)
async def test_deep_citation_invariant(make_ctx, handler, intent, nct):
    s = synthetic_study(3) if nct == "synthetic" else Study(load_study_raw(nct))
    res = await _run(handler, make_ctx, s, intent)
    assert _items(res.visualization)
    _check_all_citations(res, s.raw)


# == (5) the analysis LLM's rationale never reaches the output =====================================

SENTINEL = "ZZ-RATIONALE-SENTINEL-7f3a91 the model says pick these"


class RecordingStub(StubLLM):
    """Answers every analysis call with a VALID pick (first catalog measure / all groups) + the sentinel."""

    def __init__(self):
        self.calls = 0
        super().__init__({"analysis": self._answer})

    def _answer(self, user: str, schema):
        self.calls += 1
        cat = json.loads(user)
        return LLMPick(
            purpose=cat["purpose"],
            measure_indices=[m["index"] for m in cat["measures"][:1]],
            group_ids=[g["id"] for g in cat["groups"]],
            rationale=SENTINEL,
        )


def _serialize(res) -> str:
    return json.dumps(
        {
            "viz": res.visualization.model_dump(mode="json", exclude_none=True),
            "notes": [n.model_dump(mode="json") for n in res.notes],
            "alternatives": [a.model_dump(mode="json") for a in res.alternatives],
            "facts": res.facts,
            "full_refs": {k: [r.to_citation().model_dump(mode="json") for r in v] for k, v in res.full_refs.items()},
        },
        default=str,
    )


@pytest.mark.parametrize("nct", ALL + ["synthetic5"])
@pytest.mark.parametrize("handler,intent", HANDLERS, ids=HANDLER_IDS)
async def test_llm_rationale_not_in_output(make_ctx, handler, intent, nct):
    s = synthetic_study(5) if nct == "synthetic5" else Study(load_study_raw(nct))
    llm = RecordingStub()
    res = await _run(handler, make_ctx, s, intent, query="what happened", llm=llm)
    blob = _serialize(res)
    assert SENTINEL not in blob and "RATIONALE-SENTINEL" not in blob


async def test_llm_rationale_stub_is_actually_used(make_ctx):
    """Guard for the test above: the stub is consulted and its (valid) pick accepted on a 5-arm study."""
    s = synthetic_study(5)
    for handler, intent in [
        (R.results_outcomes, Intent.RESULTS_OUTCOMES),
        (R.risks_adverse_events, Intent.RISKS_ADVERSE_EVENTS),
        (R.dropout_flow, Intent.DROPOUT_FLOW),
    ]:
        llm = RecordingStub()
        res = await _run(handler, make_ctx, s, intent, query="what happened", llm=llm)
        assert llm.calls == 1, intent
        assert res.visualization.meta.diagnostics["analysis"]["method"] == "llm", intent
