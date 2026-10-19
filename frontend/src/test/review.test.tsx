// Regression tests for the code-review findings (numbered as in the review).
import { afterEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { SAMPLES } from "../samples";
import { LIGHT } from "../theme";
import { categories, formatChannel, safeHttpUrl, tooltipLines } from "../format";
import { CitationPanel } from "../components/CitationPanel";
import { ResponseView } from "../components/ResponseView";
import { Renderer, RENDERERS } from "../viz/Renderer";
import {
  datumTooltip, forestOption, geoCountryOption, geoSiteOption, histogramBins, histogramOption, logFallbacks,
  survivalOption, timelineGroups, timelineOption,
} from "../viz/options";
import App from "../App";
import { toRenderViz } from "../adapt";
import optionsSrc from "../viz/options.ts?raw";
import plainSrc from "../viz/Plain.tsx?raw";
import rendererSrc from "../viz/Renderer.tsx?raw";
import { RESERVED_KEYS, type Datum, type Visualization, type VisualizeResponse } from "../types";

vi.mock("echarts-for-react/lib/core", () => ({ default: () => <div data-testid="echart" /> }));

const sample = (name: string): VisualizeResponse => structuredClone(SAMPLES.find((s) => s.name === name)!.response);
const vizOf = (name: string): Visualization => toRenderViz(sample(name))!;
const rows = (v: Visualization) => v.data as Datum[];
type Opt = Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any

afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); });

describe("1. safe links", () => {
  it("allows only http(s) URLs", () => {
    expect(safeHttpUrl("https://clinicaltrials.gov/study/NCT03548935")).toBe("https://clinicaltrials.gov/study/NCT03548935");
    expect(safeHttpUrl("http://example.org/x")).toBe("http://example.org/x");
    expect(safeHttpUrl("javascript:alert(1)")).toBeUndefined();
    expect(safeHttpUrl("data:text/html,hi")).toBeUndefined();
    expect(safeHttpUrl("/relative")).toBeUndefined();
    expect(safeHttpUrl("")).toBeUndefined();
  });

  it("renders unsafe citation URLs and URL-less derived counts as text", () => {
    const datum: Datum = {
      id: "d1",
      citations: [
        { nct_id: "NCT03548935", excerpt: "x", field_path: "a.b", kind: "field_value", url: "javascript:alert(1)" },
        { nct_id: "", excerpt: "12", field_path: "$.totalCount", kind: "derived_count", url: "" },
      ],
    };
    const { container } = render(<CitationPanel datum={datum} title="t" requestId="r" onClose={() => {}} />);
    const hrefs = [...container.querySelectorAll("a")].map((a) => a.getAttribute("href"));
    expect(hrefs.every((h) => h?.startsWith("https://"))).toBe(true);
    expect(hrefs).toContain("https://clinicaltrials.gov/study/NCT03548935"); // falls back to the study page
    expect(screen.getByText("Count from ClinicalTrials.gov")).toBeTruthy();
  });
});

describe("2. citation panel state", () => {
  const truncated: Datum = {
    id: "d1", citations: [{ nct_id: "NCT03548935", excerpt: "a", field_path: "p", kind: "field_value", url: "https://clinicaltrials.gov/study/NCT03548935" }],
    citations_total: 3,
  };

  it("hides \"Show all\" after loading the full list", async () => {
    const full = [0, 1, 2].map((i) => ({ ...truncated.citations![0], excerpt: `e${i}` }));
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ citations: full }), { status: 200 })));
    render(<CitationPanel datum={truncated} title="t" requestId="r" onClose={() => {}} />);
    fireEvent.click(screen.getByRole("button", { name: "Show all 3 excerpts" }));
    await waitFor(() => expect(screen.getByText("e2")).toBeTruthy());
    expect(screen.queryByText(/^Show all/)).toBeNull();
  });

  it("ignores a stale fetch after switching datums (panel keyed by datum id)", async () => {
    let resolve!: (r: Response) => void;
    vi.stubGlobal("fetch", vi.fn((_u: string, init?: RequestInit) => new Promise<Response>((res, rej) => {
      resolve = res;
      init?.signal?.addEventListener("abort", () => rej(Object.assign(new Error("aborted"), { name: "AbortError" })));
    })));
    const r = sample("bar_chart_stacked");
    const [a, b] = rows(toRenderViz(r)!);
    const onClose = () => {};
    const { rerender } = render(<CitationPanel key={a.id} datum={a} title="A" requestId={r.meta.request_id} onClose={onClose} />);
    fireEvent.click(screen.getByRole("button", { name: /excerpts$/ }));
    rerender(<CitationPanel key={b.id} datum={b} title="B" requestId={r.meta.request_id} onClose={onClose} />);
    await act(async () => resolve(new Response(JSON.stringify([{ ...a.citations![0], excerpt: "STALE" }]), { status: 200 })));
    expect(screen.queryByText("STALE")).toBeNull();
    expect(screen.getByRole("heading", { name: "B" })).toBeTruthy();
    expect(screen.getByRole("button", { name: /excerpts$/ })).toBeTruthy();
  });
});

describe("3. geo via meta.geo", () => {
  it("country level reads region_field / value_field", () => {
    const v = vizOf("geo_map_country");
    v.encoding = { tooltip: [] }; // fields must come from meta.geo, not encoding
    const o = geoCountryOption(v, LIGHT) as Opt;
    expect(o.yAxis.data[0]).toBe("United States");
    expect(o.series[0].data[0].value).toBe(412);
  });

  it("site level reads lat_field / lon_field", () => {
    const v = vizOf("geo_map_site");
    const o = geoSiteOption(v, LIGHT) as Opt;
    const pts = o.series.flatMap((s: Opt) => s.data);
    expect(pts[0].value).toEqual([rows(v)[0].lon, rows(v)[0].lat]);
  });

  it("without meta.geo shows a warning and the data table", () => {
    const v = vizOf("geo_map_country");
    delete v.meta!.geo;
    const { container } = render(<Renderer viz={v} onSelect={() => {}} />);
    expect(screen.getByText(/cannot be drawn/)).toBeTruthy();
    expect(container.querySelector(".data-table-details")).not.toBeNull();
  });
});

describe("4. timeline lanes", () => {
  it("groups lanes by axis kind with a separate x axis each", () => {
    const v = vizOf("timeline_relative");
    const groups = timelineGroups(v);
    expect(groups.map((g) => g.kind)).toEqual(["calendar", "relative"]);
    expect(groups[1].lanes.map((l) => l.name)).toEqual(["Treatment", "Visits"]);
    const o = timelineOption(v, LIGHT) as Opt;
    expect(o.grid).toHaveLength(2);
    expect(o.xAxis.map((a: Opt) => a.type)).toEqual(["time", "value"]);
    expect(o.xAxis[1].name).toBe("Days from Randomization");
    // a lane without end_field draws point markers (end == start flag)
    const visits = o.series.find((s: Opt) => s.name.startsWith("Visits"));
    expect(visits.data.every((d: Opt) => d.value[3] === 1)).toBe(true);
  });

  it("calendar lanes from the sample use start/end fields named in meta.lanes", () => {
    const v = vizOf("timeline");
    const o = timelineOption(v, LIGHT) as Opt;
    expect(o.xAxis).toHaveLength(1);
    const ongoing = o.series.flatMap((s: Opt) => s.data).find((d: Opt) => d.datumId === "d4");
    expect(ongoing.value[3]).toBe(1); // no end date -> point / ongoing
  });
});

describe("5. missing numbers are skipped, never 0", () => {
  it("survival", () => {
    const v = vizOf("survival_curve");
    delete rows(v)[2].survival;
    const o = survivalOption(v, LIGHT) as Opt;
    const ids = o.series.filter((s: Opt) => s.type === "scatter").flatMap((s: Opt) => s.data.map((d: Opt) => d.datumId));
    expect(ids).not.toContain(rows(v)[2].id);
    const ys = o.series.filter((s: Opt) => s.type === "scatter").flatMap((s: Opt) => s.data.map((d: Opt) => d.value[1]));
    expect(ys.some((y: number) => y === 0)).toBe(false);
  });

  it("histogram, including open-ended bins", () => {
    const v = vizOf("histogram");
    delete rows(v)[0].count;
    const bins = histogramBins(v);
    expect(bins.map((b) => b.d.id)).not.toContain(rows(v)[0].id);
    const open = bins.find((b) => b.open)!;
    expect(open.end).toBeGreaterThan(open.start);
    expect(((histogramOption(v, LIGHT) as Opt).series[0].data as Opt[]).length).toBe(rows(v).length - 1);
  });
});

describe("6. log axes", () => {
  it("uses log when all values are positive", () => {
    const v = vizOf("forest_plot");
    expect(logFallbacks(v)).toEqual([]);
    expect((forestOption(v, LIGHT) as Opt).xAxis.type).toBe("log");
  });

  it("falls back to linear when a value is <= 0 (no developer hint on screen)", () => {
    const v = vizOf("forest_plot");
    rows(v)[0].ci_low = 0;
    expect((forestOption(v, LIGHT) as Opt).xAxis.type).toBe("value");
    render(<Renderer viz={v} onSelect={() => {}} />);
    expect(screen.queryByText(/linear scale/)).toBeNull();
  });
});

describe("7. accessibility", () => {
  it("data table under charts is keyboard reachable and opens the panel, which takes focus", () => {
    const r = sample("bar_chart_grouped");
    render(<ResponseView response={r} onAlternative={() => {}} busy={false} />);
    const btn = screen.getAllByRole("button", { name: /^Show 1 source for/ })[0];
    fireEvent.click(btn);
    const panel = screen.getByRole("complementary", { name: "Sources" });
    expect(document.activeElement).toBe(panel);
  });

  it("Escape closes the panel only when it has focus", () => {
    const r = sample("kpi_cards");
    render(<ResponseView response={r} onAlternative={() => {}} busy={false} />);
    fireEvent.click(document.querySelector(".kpi-card")!);
    fireEvent.keyDown(window, { key: "Escape" });
    fireEvent.keyDown(document.body, { key: "Escape" });
    expect(screen.queryByRole("complementary", { name: "Sources" })).not.toBeNull();
    fireEvent.keyDown(screen.getByRole("complementary", { name: "Sources" }), { key: "Escape" });
    expect(screen.queryByRole("complementary", { name: "Sources" })).toBeNull();
  });
});

describe("8. documented keys only", () => {
  it("no sample uses a reserved key of its own type as a channel field", () => {
    // Most reserved keys only mean something for one type (e.g. `status` for eligibility_checklist), so a
    // phase x status heatmap may encode a `status` field.
    const scoped: Record<string, string> = {
      is_gap: "heatmap", source: "survival_curve", stage: "sankey", kind: "kpi_cards",
      section: "eligibility_checklist", status: "eligibility_checklist", evaluated_by: "eligibility_checklist", parent_text: "eligibility_checklist",
    };
    for (const { response } of SAMPLES) {
      if (!response.visualization) continue;
      const { type, encoding: e } = response.visualization;
      const fields = [e.x, e.x2, e.y, e.y2, e.series, e.size, e.label, e.error_low, e.error_high].filter(Boolean).map((c) => c!.field);
      const reserved = (RESERVED_KEYS as readonly string[]).filter((k) => !scoped[k] || scoped[k] === type);
      for (const f of fields) expect(reserved.includes(f), `${response.meta.request_id}: ${f}`).toBe(false);
    }
  });

  it("renderer source reads no undocumented data keys", () => {
    // Datum property reads like d.foo / r.foo / n.foo must be reserved keys.
    const reads = [optionsSrc, plainSrc, rendererSrc].join("\n").match(/\b(?:d|r|n|e|row|datum)\.(?!id\b|citations\b)([a-z_]+)\b/g) ?? [];
    const allowed = new Set<string>([...RESERVED_KEYS, "length", "data", "nodes", "edges"]);
    const bad = reads.map((m) => m.split(".")[1]).filter((k) => !allowed.has(k));
    expect(bad).toEqual([]);
  });

  it("renders only scalar summary/stats entries", () => {
    const v = vizOf("histogram");
    v.meta!.summary = { n: 692, nested: { a: 1 }, list: [1, 2] };
    render(<Renderer viz={v} onSelect={() => {}} />);
    expect(screen.getByText("N")).toBeTruthy();
    expect(screen.getByText("List")).toBeTruthy();
    expect(screen.queryByText("Nested")).toBeNull();
  });
});

describe("9. sorting and tooltip formatting", () => {
  const data: Datum[] = [{ id: "a", k: "x", v: 3 }, { id: "b", k: "y", v: 1 }, { id: "c", k: "z", v: 2 }];
  it("sort.by defaults to ascending; order desc reverses", () => {
    expect(categories(data, "k", { by: "v" })).toEqual(["y", "z", "x"]);
    expect(categories(data, "k", { by: "v", order: "desc" })).toEqual(["x", "z", "y"]);
  });

  it("tooltip fields use the matching channel's format and unit", () => {
    const enc = { y: { field: "v", type: "quantitative" as const, title: "Rate", format: ".1%" }, tooltip: ["v"] };
    expect(tooltipLines({ id: "a", v: 0.123 }, ["v"], undefined, enc)).toEqual(["Rate: 12.3%"]);
    const v: Visualization = { type: "bar_chart", title: "t", encoding: { x: { field: "k", type: "nominal", title: "K" }, ...enc }, data: data };
    expect(datumTooltip({ id: "a", k: "x", v: 0.5 }, v)).toContain("50.0%");
    expect(formatChannel({ id: "a", year: "2019-06-01" }, { field: "year", type: "temporal", title: "Y", time_format: "%Y" })).toBe("2019");
  });
});

describe("10. robustness", () => {
  it("error boundary shows raw JSON when a renderer crashes", () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(console, "info").mockImplementation(() => {});
    const original = RENDERERS.histogram;
    RENDERERS.histogram = () => { throw new Error("boom"); };
    try {
      const v = vizOf("histogram");
      render(<Renderer viz={v} onSelect={() => {}} />);
      expect(screen.getByRole("alert").textContent).toContain("boom");
      expect(screen.getByText("Raw visualization JSON")).toBeTruthy();
      expect(document.querySelector(".raw-json")!.textContent).toContain('"bin_start"');
    } finally {
      RENDERERS.histogram = original;
    }
  });

  it("shows a Retry-After hint for 503", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("{}", { status: 503, headers: { "Retry-After": "30" } })));
    render(<App />);
    fireEvent.change(screen.getByLabelText("Question"), { target: { value: "q" } });
    fireEvent.click(screen.getByRole("button", { name: "Visualize" }));
    await waitFor(() => expect(screen.getByText("Service unavailable")).toBeTruthy());
    expect(screen.getByText("Try again in 30 s.")).toBeTruthy();
  });

  it("chart tooltips keep the theme styling", () => {
    const o = histogramOption(vizOf("histogram"), LIGHT) as Opt;
    expect(o.tooltip.backgroundColor).toBe(LIGHT.surface);
    expect(typeof o.tooltip.formatter).toBe("function");
  });
});
