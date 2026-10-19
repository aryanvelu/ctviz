// Public JSON 2.0: status branches, flattened meta hints, nct_ids / query_url in the citation panel,
// site-level geo, ego networks, the Response JSON panel, and re-asking from chips.
import { afterEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { SAMPLES } from "../samples";
import { LIGHT } from "../theme";
import { hintsOf, toRenderViz } from "../adapt";
import { CitationPanel } from "../components/CitationPanel";
import { ResponseView } from "../components/ResponseView";
import { geoCountryOption, geoSiteOption, networkOption } from "../viz/options";
import App from "../App";
import type { Datum, GraphData, VisualizeResponse } from "../types";

const captured = vi.hoisted(() => ({ charts: [] as { option: Record<string, unknown>; onEvents?: Record<string, (p: unknown) => void> }[] }));
vi.mock("echarts-for-react/lib/core", () => ({
  default: (props: { option: Record<string, unknown>; onEvents?: Record<string, (p: unknown) => void> }) => {
    captured.charts.push({ option: props.option, onEvents: props.onEvents });
    return <div data-testid="echart" />;
  },
}));

const sample = (name: string): VisualizeResponse => structuredClone(SAMPLES.find((s) => s.name === name)!.response);
type Opt = Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any

afterEach(() => { captured.charts = []; vi.unstubAllGlobals(); vi.restoreAllMocks(); });

function mockFetch(body: unknown) {
  const fn = vi.fn(async () => new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } }));
  vi.stubGlobal("fetch", fn);
  return fn;
}
const sentBodies = (fn: ReturnType<typeof mockFetch>) =>
  fn.mock.calls.map((c) => JSON.parse(String((c as unknown as [string, RequestInit])[1].body)));

describe("status branches", () => {
  it("needs_clarification: clarification card, unmatched terms, suggestion chips, no chart", () => {
    const r = sample("live_needs_clarification");
    const onSuggest = vi.fn();
    const { container } = render(<ResponseView response={r} onSuggest={onSuggest} />);
    expect(container.querySelector("figure.viz")).toBeNull();
    expect(captured.charts).toHaveLength(0);
    const card = screen.getByRole("region", { name: "Clarification" });
    expect(within(card).getByText(r.clarification!.message)).toBeTruthy();
    expect(within(card).getByText("flurbozzle")).toBeTruthy();
    fireEvent.click(within(card).getByRole("button", { name: r.clarification!.suggestions![0] }));
    expect(onSuggest).toHaveBeenCalledWith(r.clarification!.suggestions![0]);
    expect(screen.getByTestId("status").textContent).toContain("Needs clarification");
  });

  it("unanswerable: explains why, no chart", () => {
    const r = sample("live_unanswerable");
    const { container } = render(<ResponseView response={r} />);
    expect(container.querySelector("figure.viz")).toBeNull();
    expect(screen.getByText("The registry cannot answer this kind of question")).toBeTruthy();
    expect(container.querySelector(".clarification.unanswerable")).not.toBeNull();
  });

  it("trial_required: candidate trials are rows that re-ask with that nct_id", () => {
    const r = sample("trial_required");
    const onPick = vi.fn();
    render(<ResponseView response={r} onPickTrial={onPick} />);
    fireEvent.click(screen.getByRole("button", { name: /NCT02142738/ }));
    expect(onPick).toHaveBeenCalledWith("NCT02142738");
  });

  it("ok: chart, then summary and follow-up chips", () => {
    const r = sample("live_time_series");
    const onAsk = vi.fn();
    const { container } = render(<ResponseView response={r} onAsk={onAsk} />);
    expect(container.querySelector('figure.viz[data-viz-type="time_series"]')).not.toBeNull();
    expect(screen.queryByRole("region", { name: "Clarification" })).toBeNull();
    const q = r.meta.follow_ups![0];
    fireEvent.click(screen.getByRole("button", { name: q }));
    expect(onAsk).toHaveBeenCalledWith(q);
  });
});

describe("flattened meta hints", () => {
  it("gathers hints from response meta into viz.meta for the renderer", () => {
    const r = sample("live_geo_map_country");
    expect((r.visualization as unknown as Record<string, unknown>).meta).toBeUndefined();
    const viz = toRenderViz(r)!;
    expect(viz.meta?.geo).toEqual(r.meta.geo);
    expect(viz.meta?.units).toEqual(r.meta.units);
    expect(viz.meta).not.toHaveProperty("coverage");
    expect(viz.meta).not.toHaveProperty("filters");
  });

  it("approximation present means is_approximation; chart_diagnostics come from meta.debug", () => {
    const h = hintsOf({ request_id: "r", created_at: "t", approximation: { method: "digitized" }, debug: { chart_diagnostics: { picked: "OS" } } });
    expect(h.approximation).toEqual({ is_approximation: true, method: "digitized" });
    expect(h.diagnostics).toEqual({ picked: "OS" });
    expect(hintsOf({ request_id: "r", created_at: "t" })).toEqual({});
  });

  it("defaults omitted lists (no notes, alternatives, follow-ups, validation) without crashing", () => {
    const r = sample("live_kpi_cards");
    for (const k of ["notes", "alternatives", "follow_ups", "validation", "assumptions", "filters"] as const) delete r.meta[k];
    const { container } = render(<ResponseView response={r} />);
    expect(container.querySelector(".kpi-card")).not.toBeNull();
    expect(container.querySelector(".notes")).toBeNull();
  });

  it("the no-chart response renders with no visualization", () => {
    expect(toRenderViz(sample("live_unanswerable"))).toBeNull();
  });
});

describe("datum panel: nct_ids and query_url", () => {
  it("clicking a point lists every trial behind it with study links", () => {
    const r = sample("live_time_series");
    render(<ResponseView response={r} />);
    const d = (r.visualization!.data as Datum[])[0];
    act(() => captured.charts[0].onEvents?.click?.({ data: { datumId: d.id } }));
    const panel = screen.getByRole("complementary", { name: "Sources" });
    const list = within(panel).getByRole("region", { name: "Trials" });
    // Schema 2.2 lists only the first ids inline; the heading counts nct_ids_total when present.
    const total = d.nct_ids_total ?? d.nct_ids!.length;
    expect(within(list).getByText(`${total} trials`)).toBeTruthy();
    expect(list.querySelectorAll("a").length).toBe(Math.min(12, d.nct_ids!.length));
    expect(list.querySelector("a")!.getAttribute("href")).toBe(`https://clinicaltrials.gov/study/${d.nct_ids![0]}`);
    if (total === d.nct_ids!.length && total > 12) {
      fireEvent.click(within(list).getByRole("button", { name: `Show all ${total}` }));
      expect(list.querySelectorAll("a").length).toBe(total);
    }
  });

  it("\"Show all\" fetches the complete trial list when only the first ids came inline (one request fills both)", async () => {
    const first = Array.from({ length: 20 }, (_, i) => `NCT${String(10000000 + i)}`);
    const complete = Array.from({ length: 57 }, (_, i) => `NCT${String(10000000 + i)}`);
    const cite = (nct: string, i: number) => ({ nct_id: nct, excerpt: `x${i}`, field_path: "protocolSection.statusModule.overallStatus", url: `https://clinicaltrials.gov/study/${nct}` });
    const datum: Datum = {
      id: "d1", trial_count: 57, nct_ids: first, nct_ids_total: 57,
      citations: first.slice(0, 3).map(cite), citations_total: 57,
    };
    const fn = mockFetch({ citations: complete.map(cite), nct_ids: complete, nct_ids_total: 57 });
    render(<CitationPanel datum={datum} title="Recruiting" requestId="req-1" onClose={() => {}} />);
    const list = screen.getByRole("region", { name: "Trials" });
    expect(within(list).getByText("57 trials")).toBeTruthy();
    expect(list.querySelectorAll("a").length).toBe(12);
    fireEvent.click(within(list).getByRole("button", { name: "Show all 57" }));
    await waitFor(() => expect(list.querySelectorAll("a").length).toBe(57));
    expect(fn).toHaveBeenCalledTimes(1);
    expect(String((fn.mock.calls[0] as unknown as [string])[0])).toBe("/v1/responses/req-1/datums/d1/citations");
    expect(within(list).getByText("57 trials")).toBeTruthy();
    expect(within(list).queryByText(/Listing the first/)).toBeNull();
    // The same response filled the excerpts: no second request and no "Show all … excerpts" left.
    expect(screen.getByText("x56")).toBeTruthy();
    expect(screen.queryByRole("button", { name: /excerpts$/ })).toBeNull();
  });

  it("shows the registry search behind a registry_counts value", () => {
    const datum: Datum = {
      id: "d1", trial_count: 42, query_url: "https://clinicaltrials.gov/search?intr=semaglutide&aggFilters=phase:3",
      citations: [{ excerpt: "42", field_path: "$.totalCount", kind: "derived_count", url: "https://clinicaltrials.gov/search?intr=semaglutide&aggFilters=phase:3" }],
    };
    render(<CitationPanel datum={datum} title="Phase 3" requestId="r" onClose={() => {}} />);
    const link = screen.getByRole("link", { name: "open this search" });
    expect(link.getAttribute("href")).toBe(datum.query_url);
    expect(screen.getByText(/^Count from ClinicalTrials.gov/)).toBeTruthy();
    expect(document.querySelectorAll("li.citation")).toHaveLength(0); // the total is not repeated as an excerpt
    expect(screen.queryByText(/totalCount|derived/)).toBeNull();
  });
});

describe("geo_map", () => {
  it("site level: points at lat/lon, sized by encoding.size, axes fitted to the points", () => {
    const viz = toRenderViz(sample("live_geo_map_site"))!;
    const o = geoSiteOption(viz, LIGHT) as Opt;
    const pts = o.series.flatMap((s: Opt) => s.data);
    const rows = viz.data as Datum[];
    const first = rows.find((r) => r.lat !== undefined)!;
    expect(pts.find((p: Opt) => p.datumId === first.id).value).toEqual([first.lon, first.lat]);
    const sizes = new Set(pts.map((p: Opt) => p.symbolSize));
    expect(sizes.size).toBeGreaterThan(1);
    const biggest = rows.reduce((a, b) => (Number(b.trial_count) > Number(a.trial_count) ? b : a));
    expect(pts.find((p: Opt) => p.datumId === biggest.id).symbolSize).toBe(Math.max(...sizes as Set<number>));
    expect(o.xAxis.min).toBeGreaterThan(-180); // US sites: not the whole world
    expect(o.xAxis.max).toBeLessThan(180);
  });

  it("country level falls back to encoding.x / encoding.y when meta.geo names no fields", () => {
    const viz = toRenderViz(sample("live_geo_map_country"))!;
    viz.meta = { ...viz.meta, geo: { level: "country" } };
    const o = geoCountryOption(viz, LIGHT) as Opt;
    const rows = viz.data as Datum[];
    const top = [...rows].sort((a, b) => Number(b.trial_count) - Number(a.trial_count))[0];
    expect(o.yAxis.data[0]).toBe(top.country);
  });
});

describe("network_graph", () => {
  it("ego layout puts the pinned node in the center and every other node around it", () => {
    const viz = toRenderViz(sample("live_network_ego"))!;
    expect(viz.meta?.network_layout).toBe("ego");
    const o = networkOption(viz, LIGHT) as Opt;
    expect(o.series[0].layout).toBe("none");
    const nodes = o.series[0].data as Opt[];
    const pinned = nodes.find((n) => n.id === viz.meta!.pinned![0])!;
    expect([pinned.x, pinned.y]).toEqual([0, 0]);
    for (const n of nodes) if (n !== pinned) expect(Math.hypot(n.x, n.y)).toBeGreaterThan(100);
  });

  it("precomputed force layout otherwise (fixed positions); node size and edge width follow the encoded fields", () => {
    const viz = toRenderViz(sample("live_network_force"))!;
    const o = networkOption(viz, LIGHT) as Opt;
    expect(o.series[0].layout).toBe("none");
    expect(o.series[0].force).toBeUndefined();
    const g = viz.data as GraphData;
    expect(new Set((o.series[0].links as Opt[]).map((l) => l.lineStyle.width)).size).toBeGreaterThan(1);
    expect(o.legend.data.length).toBeGreaterThan(0);
    expect(o.series[0].data).toHaveLength(g.nodes.length);
  });
});

describe("response JSON panel", () => {
  it("one collapsed <details> with the body as received; no validation, interpretation or debug UI", () => {
    const r = sample("trial_required");
    const raw = JSON.stringify(r);
    const { container } = render(<ResponseView response={r} raw={raw} />);
    const details = container.querySelector("details.json-panel") as HTMLDetailsElement;
    expect(details.open).toBe(false);
    expect(within(details).getByText("Response JSON")).toBeTruthy();
    expect(screen.getByTestId("public-json").textContent).toBe(JSON.stringify(JSON.parse(raw), null, 2));
    expect(within(details).getByRole("button", { name: "Copy" })).toBeTruthy();
    expect(screen.queryByRole("tab")).toBeNull();
    expect(screen.queryByText(/Validation/)).toBeNull();
    expect(screen.queryByText(/Interpretation/)).toBeNull();
    expect(container.querySelectorAll("details").length).toBe(1);
  });

  it("status badge only when status != ok; footer shows the data timestamp", () => {
    const ok = sample("live_time_series");
    ok.meta.data_timestamp = "2026-10-17";
    const { container, unmount } = render(<ResponseView response={ok} />);
    expect(screen.queryByTestId("status")).toBeNull();
    expect(container.querySelector(".footer-meta")!.textContent).toBe("Data from ClinicalTrials.gov as of 2026-10-17");
    unmount();
    render(<ResponseView response={sample("live_unanswerable")} />);
    expect(screen.getByTestId("status").textContent).toBe("Unanswerable");
  });
});

describe("App re-asks from chips", () => {
  it("sends drugs as drug_names (never include_trace), then re-asks a follow-up", async () => {
    const r = sample("live_time_series");
    const fn = mockFetch(r);
    render(<App />);
    fireEvent.change(screen.getByLabelText("Question"), { target: { value: "Compare phases" } });
    fireEvent.change(screen.getByLabelText("Drug(s)"), { target: { value: "semaglutide, tirzepatide" } });
    fireEvent.click(screen.getByRole("button", { name: "Visualize" }));
    await waitFor(() => expect(screen.getByRole("button", { name: r.meta.follow_ups![0] })).toBeTruthy());
    fireEvent.click(screen.getByRole("button", { name: r.meta.follow_ups![0] }));
    await waitFor(() => expect(fn).toHaveBeenCalledTimes(2));
    const [first, second] = sentBodies(fn);
    expect(first).toMatchObject({ query: "Compare phases", drug_names: ["semaglutide", "tirzepatide"] });
    expect(first.drug_name).toBeUndefined();
    expect(first.include_trace).toBeUndefined();
    expect(second.query).toBe(r.meta.follow_ups![0]);
  });

  it("picking a candidate trial re-asks the same question with nct_id", async () => {
    const fn = mockFetch(sample("trial_required"));
    render(<App />);
    fireEvent.change(screen.getByLabelText("Question"), { target: { value: "Am I eligible for this trial?" } });
    fireEvent.click(screen.getByRole("button", { name: "Visualize" }));
    const row = await screen.findByRole("button", { name: /^NCT03548935 Research/ });
    fireEvent.click(row);
    await waitFor(() => expect(fn).toHaveBeenCalledTimes(2));
    expect(sentBodies(fn)[1]).toMatchObject({ query: "Am I eligible for this trial?", nct_id: "NCT03548935" });
  });
});

describe("clean answer view", () => {
  it("never renders meta.notes, chart chrome hints, or the 'View as' explanations", () => {
    const r = sample("live_network_force");
    r.meta.notes = [{ code: "REGISTRY_COUNTS", severity: "info", message: "Counts are registry totals." }, { code: "TRUNCATED", severity: "warning", message: "Top 28 shown." }];
    r.meta.alternatives = [{ type: "table", reason: "Table: every value as a row" }];
    const { container } = render(<ResponseView response={r} />);
    const text = container.querySelector(".response-main > :not(details)")?.closest(".response-main")?.textContent ?? "";
    const visible = text.replace(container.querySelector(".json-panel")!.textContent ?? "", "");
    for (const s of ["Counts are registry totals", "REGISTRY COUNTS", "Top 28", "omitted", "Lead sponsor runs", "hidden", "Click any", "every value as a row"]) {
      expect(visible).not.toContain(s);
    }
    expect(screen.getByRole("button", { name: "Table" }).getAttribute("title")).toBe("Table: every value as a row");
  });

  it("follow-ups render under 'Ask next' and clicking one asks it", () => {
    const r = sample("live_time_series");
    r.meta.follow_ups = ["Which sponsors run these trials?"];
    const onAsk = vi.fn();
    render(<ResponseView response={r} onAsk={onAsk} />);
    const section = screen.getByRole("region", { name: "Follow-up questions" });
    expect(within(section).getByRole("heading", { name: "Ask next" })).toBeTruthy();
    fireEvent.click(within(section).getByRole("button", { name: /Which sponsors run these trials\?/ }));
    expect(onAsk).toHaveBeenCalledWith("Which sponsors run these trials?");
  });
});

describe("kpi cards", () => {
  it("long text values use body size with the full text in a tooltip; enums and units read as words", () => {
    const r = sample("live_kpi_cards");
    const rowsK = r.visualization!.data as Datum[];
    rowsK.push({ id: "dx", label: "Participants", value: 304, unit: "participants", detail: "ACTUAL", kind: "number" });
    const { container } = render(<ResponseView response={r} />);
    const cards = [...container.querySelectorAll(".kpi-card")] as HTMLElement[];
    const title = cards.find((c) => c.querySelector(".kpi-label")!.textContent === "Title")!;
    const value = title.querySelector(".kpi-value")!;
    expect(value.classList.contains("kpi-text")).toBe(true);
    expect(value.getAttribute("title")).toBe(rowsK[0].value);
    const status = cards.find((c) => c.querySelector(".kpi-label")!.textContent === "Status")!;
    expect(status.querySelector(".kpi-value")!.textContent).toBe("Completed");
    expect(status.querySelector(".kpi-value")!.classList.contains("kpi-text")).toBe(false);
    const enroll = cards.find((c) => c.querySelector(".kpi-label")!.textContent === "Participants")!;
    expect(enroll.querySelector(".kpi-sub")!.textContent).toBe("participants · Actual");
    expect(container.textContent).not.toMatch(/Unit:|Detail:|\d+ sources?/);
    expect(enroll.querySelector(".kpi-cites")!.textContent).toBe("Sources");
  });
});
