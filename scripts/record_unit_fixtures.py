"""Record raw ClinicalTrials.gov JSON used by unit tests (politely: 1 request / 2 s, retry on 429)."""

from __future__ import annotations

import gzip
import json
import pathlib
import sys
import time

import httpx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ctviz.ctgov.fields import FIELD_SETS, FieldSet  # noqa: E402

B = "https://clinicaltrials.gov/api/v2/studies"
ROOT = pathlib.Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "ctgov"
STUDIES = {
    "NCT03548935": "STEP 1 semaglutide obesity, phase 3, results (AE, flow, baseline, outcomes)",
    "NCT02578680": "KEYNOTE-189 pembrolizumab NSCLC, OS/PFS hazard ratios",
    "NCT02142738": "KEYNOTE-024 pembrolizumab vs chemo, survival",
    "NCT01844505": "CheckMate 067 nivolumab+ipilimumab combination melanoma",
    "NCT07654361": "Recruiting phase 3 obesity trial, no results",
    "NCT01284335": "Phase 1 dose escalation solid tumors, results",
}
COHORTS = {
    "pembrolizumab_since2015": {
        "query.intr": "Pembrolizumab",
        "filter.advanced": "AREA[StartDate]RANGE[2015-01-01,MAX]",
    },
    "obesity_since2020": {"query.cond": "obesity", "filter.advanced": "AREA[StartDate]RANGE[2020-01-01,MAX]"},
    "semaglutide": {"query.intr": "semaglutide"},
    "tirzepatide": {"query.intr": "tirzepatide"},
    "melanoma_ph2_3": {"query.cond": "melanoma", "filter.advanced": "AREA[Phase](PHASE2 OR PHASE3)"},
    "t2d_recruiting": {"query.cond": "type 2 diabetes", "filter.overallStatus": "RECRUITING"},
}


def get(client: httpx.Client, url: str, params: dict | None = None) -> dict:
    for attempt in range(6):
        r = client.get(url, params=params)
        if r.status_code == 200:
            time.sleep(2)
            return r.json()
        wait = 15 * (attempt + 1)
        print(f"  {r.status_code}; waiting {wait}s", flush=True)
        time.sleep(wait)
    raise SystemExit(f"failed: {url}")


def main() -> None:
    (ROOT / "studies").mkdir(parents=True, exist_ok=True)
    (ROOT / "cohorts").mkdir(parents=True, exist_ok=True)
    manifest: dict = {}
    fields = ",".join(dict.fromkeys(FIELD_SETS[FieldSet.PEOPLE] + FIELD_SETS[FieldSet.OUTCOMES]))
    with httpx.Client(timeout=60) as c:
        for nct, desc in STUDIES.items():
            path = ROOT / "studies" / f"{nct}.json"
            if not path.exists():
                path.write_text(json.dumps(get(c, f"{B}/{nct}")))
            d = json.loads(path.read_text())
            manifest[nct] = {"description": desc, "hasResults": d.get("hasResults")}
        for name, q in COHORTS.items():
            print("cohort", name, flush=True)
            p = {**q, "fields": fields, "pageSize": "400", "countTotal": "true", "sort": "LastUpdatePostDate:desc"}
            d = get(c, B, p)
            with gzip.open(ROOT / "cohorts" / f"{name}.json.gz", "wt") as fh:
                json.dump({"params": q, "totalCount": d.get("totalCount"), "studies": d["studies"]}, fh)
            manifest[name] = {"params": q, "totalCount": d.get("totalCount"), "n": len(d["studies"])}
    (ROOT / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
