// Every sample must be the public JSON (schema 2.2) a client receives from POST /v1/visualize: it
// validates against the JSON Schema generated from ctviz/api/public.py::PublicResponse and satisfies the
// rendering rules the schema cannot express (mirrors ctviz/viz/base.py check_encoding).
// live_*.json are real responses captured from the running backend; the rest are hand-written
// samples converted to 2.2 with the backend's own projection helpers (ctviz/api/public.py).
import { describe, expect, it } from "vitest";
import Ajv2020 from "ajv/dist/2020";
import schema from "../schema.json";
import { SAMPLES } from "../samples";
import { HINT_KEYS, allDatums, toRenderViz } from "../adapt";
import { VIZ_TYPES, isGraphData, type VisualizeResponse } from "../types";

// The documentation models mark two keys required that to_public() can omit, because it drops empty
// values: PublicCitation.nct_id (absent for kind=derived_count, as its description says) and
// PublicMeta.filters (absent when nothing was searched, e.g. a clarification). Relax just those.
const responseSchema = structuredClone(schema.response) as { $defs: Record<string, { required?: string[] }> };
const optional = (def: string, key: string) => {
  const d = responseSchema.$defs[def];
  d.required = (d.required ?? []).filter((k) => k !== key);
};
optional("PublicCitation", "nct_id");
optional("PublicMeta", "filters");

const ajv = new Ajv2020({ allErrors: true, strict: false });
const validate = ajv.compile(responseSchema);

function hasNull(v: unknown): boolean {
  if (v === null) return true;
  if (Array.isArray(v)) return v.some(hasNull);
  if (typeof v === "object" && v) return Object.values(v).some(hasNull);
  return false;
}

/** The server drops empty lists/objects; a sample with one would hide renderer default bugs. Exception (2.2):
 * every datum carries `citations`, so an uncited datum (one with a cite_note) has `citations: []`. */
function emptyPaths(v: unknown, path = "$", out: string[] = [], parent?: Record<string, unknown>): string[] {
  if (Array.isArray(v)) {
    const uncited = path.endsWith(".citations") && typeof parent?.cite_note === "string";
    if (!v.length && path !== "$.visualization.data" && !uncited) out.push(path);
    v.forEach((x, i) => emptyPaths(x, `${path}[${i}]`, out));
  } else if (v && typeof v === "object") {
    if (!Object.keys(v).length) out.push(path);
    for (const [k, x] of Object.entries(v)) emptyPaths(x, `${path}.${k}`, out, v as Record<string, unknown>);
  }
  return out;
}

describe("sample responses", () => {
  it("covers every visualization type and every status", () => {
    const types = new Set(SAMPLES.map((s) => s.response.visualization?.type).filter(Boolean));
    expect([...VIZ_TYPES].filter((t) => !types.has(t))).toEqual([]);
    const statuses = new Set(SAMPLES.map((s) => s.response.status));
    expect([...statuses].sort()).toEqual(["needs_clarification", "ok", "unanswerable"]);
  });

  for (const { name, response } of SAMPLES) {
    describe(name, () => {
      it("validates against the public response JSON Schema", () => {
        const ok = validate(response);
        expect(validate.errors ?? [], JSON.stringify(validate.errors, null, 2)).toEqual([]);
        expect(ok).toBe(true);
      });

      it("is the 2.2 shape: no nulls, no empty containers, hints flattened into meta", () => {
        expect(hasNull(response)).toBe(false);
        expect(emptyPaths(response)).toEqual([]);
        expect(response.meta.schema_version).toBe("2.2");
        expect(response.meta.request_id).toBeTruthy();
        expect((response.visualization as Record<string, unknown> | undefined)?.meta).toBeUndefined();
      });

      it("has a chart only when status is ok, and a clarification otherwise", () => {
        const r = response as VisualizeResponse;
        if (r.status === "ok") {
          expect(r.visualization).toBeDefined();
          expect(r.clarification).toBeUndefined();
        } else {
          expect(r.visualization).toBeUndefined();
          expect(r.clarification?.message).toBeTruthy();
          for (const k of HINT_KEYS) expect(r.meta[k], `hint ${k} without a chart`).toBeUndefined();
        }
      });

      if (!response.visualization) return;
      const viz = toRenderViz(response)!;

      it("states its citation policy (2.2)", () => {
        expect(response.meta.citation_policy?.study_url).toBe("https://clinicaltrials.gov/study/{nct_id}");
      });

      it("has unique datum ids and cites every datum (or explains why not)", () => {
        const all = allDatums(viz);
        const ids = all.map((d) => d.id);
        expect(new Set(ids).size).toBe(ids.length);
        for (const d of all) {
          expect((d.citations?.length ?? 0) > 0 || typeof d.cite_note === "string", `datum ${d.id}`).toBe(true);
          if (d.citations_total !== undefined) expect(d.citations_total).toBeGreaterThanOrEqual(d.citations?.length ?? 0);
          for (const id of d.nct_ids ?? []) expect(id).toMatch(/^NCT\d{8}$/);
          if (d.nct_ids) { // 2.2: capped list (citation_policy.nct_ids_per_datum) plus the complete count
            expect(d.nct_ids.length).toBeLessThanOrEqual(20);
            expect(d.nct_ids_total, `datum ${d.id} nct_ids_total`).toBeGreaterThanOrEqual(d.nct_ids.length);
          }
          if (d.query_url !== undefined) expect(d.query_url).toMatch(/^https:\/\/clinicaltrials\.gov\//);
          for (const c of d.citations ?? []) {
            if (c.kind === "text_excerpt") {
              expect(c.char_start).toBeTypeOf("number");
              expect(c.char_end! - c.char_start!).toBe(c.excerpt.length);
              expect(c.excerpt.length).toBeLessThanOrEqual(300);
            } else if (c.kind === "derived_count") {
              expect(c.nct_id).toBeUndefined();
              expect(c.field_path ?? d.field_path).toBe("$.totalCount");
            } else {
              expect(c.kind).toBeUndefined(); // field_value is the default and is omitted
              expect(c.nct_id).toMatch(/^NCT\d{8}$/);
            }
            expect(c.url).toBeUndefined(); // 2.2: the study link is meta.citation_policy.study_url
          }
        }
      });

      it("has every encoded field present on every datum", () => {
        const enc = viz.encoding;
        if (isGraphData(viz.data)) {
          expect(["network_graph", "sankey"]).toContain(viz.type);
          const ne = { label: "label", ...enc.nodes };
          const ee = { source: "source", target: "target", weight: "weight", ...enc.edges };
          const nodeIds = new Set(viz.data.nodes.map((n) => n.id));
          for (const n of viz.data.nodes) expect(n[ne.label!], `node ${n.id}`).toBeDefined();
          for (const e of viz.data.edges) {
            expect(nodeIds.has(String(e[ee.source!])), `edge ${e.id} source`).toBe(true);
            expect(nodeIds.has(String(e[ee.target!])), `edge ${e.id} target`).toBe(true);
            if (ee.weight) expect(e[ee.weight], `edge ${e.id} weight`).toBeDefined();
          }
          if (viz.meta?.network_layout === "ego") expect(viz.meta.pinned?.length).toBeGreaterThan(0);
          return;
        }
        const fields = [enc.x, enc.y, enc.series, enc.size, enc.label].filter(Boolean).map((c) => c!.field);
        for (const c of viz.meta?.columns ?? []) fields.push(c.field);
        for (const row of viz.data) for (const f of fields) expect(row[f], `${row.id}.${f}`).toBeDefined();
      });
    });
  }
});
