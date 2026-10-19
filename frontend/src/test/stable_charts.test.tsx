// Charts must not redraw or re-animate on unrelated re-renders (e.g. clicking a datum opens the
// citation panel), and the network graph must use fixed, deterministic positions instead of a live
// force simulation.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { SAMPLES } from "../samples";
import { ResponseView } from "../components/ResponseView";
import { CitationPanel, TrialList } from "../components/CitationPanel";
import { toRenderViz } from "../adapt";
import { FORCE_MAX_NODES, forcePositions, networkOption } from "../viz/options";
import { LIGHT } from "../theme";
import { RESERVED_KEYS, type Datum, type GraphData, type Visualization, type VisualizeResponse } from "../types";

const captured = vi.hoisted(() => ({ charts: [] as { option: Record<string, unknown>; onEvents?: Record<string, (p: unknown) => void> }[] }));

vi.mock("echarts-for-react/lib/core", () => ({
  default: (props: { option: Record<string, unknown>; onEvents?: Record<string, (p: unknown) => void> }) => {
    captured.charts.push({ option: props.option, onEvents: props.onEvents });
    return <div data-testid="echart" />;
  },
}));

beforeEach(() => { captured.charts = []; });
afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); });

function sample(name: string): VisualizeResponse {
  const s = SAMPLES.find((x) => x.name === name);
  if (!s) throw new Error(`no sample ${name}`);
  return s.response;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Opt = Record<string, any>;

describe("options are stable across re-renders", () => {
  for (const name of ["live_network_force", "live_sankey", "live_time_series"]) {
    it(`${name}: clicking a datum does not hand ECharts a new option`, () => {
      const r = sample(name);
      render(<ResponseView response={r} />);
      const first = captured.charts[0];
      expect(first).toBeTruthy();
      const viz = r.visualization!;
      const id = Array.isArray(viz.data) ? viz.data[0].id : viz.data.nodes[0].id;
      act(() => first.onEvents?.click?.({ data: { datumId: id } }));
      expect(screen.getByRole("complementary", { name: "Sources" })).toBeTruthy();
      expect(captured.charts.length).toBeGreaterThan(1); // the chart did re-render...
      for (const c of captured.charts) expect(c.option).toBe(first.option); // ...with the same option object
    });
  }
});

describe("network graph layout", () => {
  const viz = toRenderViz(sample("live_network_force")) as Visualization;

  it("uses fixed positions: no live force simulation", () => {
    const series = (networkOption(viz, LIGHT) as Opt).series[0];
    expect(series.layout).toBe("none");
    expect(series.force).toBeUndefined();
    expect(series.roam).toBe(true);
    expect(series.draggable).toBe(true);
    for (const d of series.data as Opt[]) {
      expect(Number.isFinite(d.x)).toBe(true);
      expect(Number.isFinite(d.y)).toBe(true);
    }
  });

  it("is deterministic: the same input gives the same picture", () => {
    const a = (networkOption(viz, LIGHT) as Opt).series[0].data.map((d: Opt) => [d.id, d.x, d.y]);
    const b = (networkOption(viz, LIGHT) as Opt).series[0].data.map((d: Opt) => [d.id, d.x, d.y]);
    expect(b).toEqual(a);
  });

  it("spreads nodes apart and keeps linked nodes closer than unlinked ones", () => {
    const g = viz.data as GraphData;
    const pos = forcePositions(g.nodes.map((n) => n.id), g.edges.map((e) => ({ source: String(e.source), target: String(e.target) })));
    const pts = [...pos.values()];
    const dist = (a: { x: number; y: number }, b: { x: number; y: number }) => Math.hypot(a.x - b.x, a.y - b.y);
    let minD = Infinity;
    for (let i = 0; i < pts.length; i++) for (let j = i + 1; j < pts.length; j++) minD = Math.min(minD, dist(pts[i], pts[j]));
    expect(minD).toBeGreaterThan(1);
    expect(forcePositions([], []).size).toBe(0);
    const tiny = forcePositions(["a", "b", "c"], [{ source: "a", target: "b", weight: 5 }]);
    expect(dist(tiny.get("a")!, tiny.get("b")!)).toBeLessThan(dist(tiny.get("a")!, tiny.get("c")!));
  });
});

describe("huge graphs skip the force refinement", () => {
  it("above FORCE_MAX_NODES only the spiral start is used (fast, still deterministic, no overlaps)", () => {
    const n = FORCE_MAX_NODES + 100;
    const ids = Array.from({ length: n }, (_, i) => `n${i}`);
    const links = ids.slice(1).map((id, i) => ({ source: ids[i], target: id }));
    const t = performance.now();
    const a = forcePositions(ids, links);
    expect(performance.now() - t).toBeLessThan(200);
    expect(a.size).toBe(n);
    expect([...forcePositions(ids, links).entries()]).toEqual([...a.entries()]);
    expect(new Set([...a.values()].map((p) => `${p.x},${p.y}`)).size).toBe(n);
  });
});

describe("citation panel: endpoint edge cases and interpolated values", () => {
  const cite = (nct: string, i: number, verified?: boolean) => ({
    nct_id: nct, excerpt: `x${i}`, field_path: "protocolSection.statusModule.overallStatus", url: `https://clinicaltrials.gov/study/${nct}`,
    ...(verified === undefined ? {} : { verified }),
  });
  const first = Array.from({ length: 20 }, (_, i) => `NCT${10000000 + i}`);
  const datum: Datum = { id: "d1", nct_ids: first, nct_ids_total: 57, citations: [cite(first[0], 0)], citations_total: 2 };

  it("a fetch without nct_ids hides \"Show all\" and says the full list is unavailable", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ citations: [cite(first[0], 0), cite(first[1], 1)] }), { status: 200 })));
    render(<CitationPanel datum={datum} title="t" requestId="r" onClose={() => {}} />);
    const list = screen.getByRole("region", { name: "Trials" });
    fireEvent.click(within(list).getByRole("button", { name: "Show all 57" }));
    await waitFor(() => expect(within(list).getByText("Full trial list not available for this response.")).toBeTruthy());
    expect(within(list).queryByRole("button", { name: /^Show all/ })).toBeNull();
    expect(within(list).getByText("57 trials")).toBeTruthy();
    expect(list.querySelectorAll("a").length).toBe(20);
  });

  it("marks excerpts the endpoint could not re-verify", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({
      citations: [cite(first[0], 0, true), cite(first[1], 1, false)], nct_ids: first, nct_ids_total: 20,
    }), { status: 200 })));
    render(<CitationPanel datum={datum} title="t" requestId="r" onClose={() => {}} />);
    fireEvent.click(screen.getByRole("button", { name: "Show all 2 excerpts" }));
    await waitFor(() => expect(screen.getByText("x1")).toBeTruthy());
    expect(screen.getAllByText("not re-verified")).toHaveLength(1);
    expect(screen.getByText("x1").closest("p")!.textContent).toContain("not re-verified");
    expect(screen.getByText("x0").closest("p")!.textContent).not.toContain("not re-verified");
  });

  it("an interpolated value links to the reported values it lies between", () => {
    expect(RESERVED_KEYS).toContain("derived_from");
    const onSelectDatum = vi.fn();
    const interp: Datum = { id: "p5", source: "interpolated", derived_from: ["p4", "p6"], citations: [] };
    const labels: Record<string, string> = { p4: "Arm A, month 6", p6: "Arm A, month 12" };
    render(<CitationPanel datum={interp} title="Arm A, month 9" requestId="r" onClose={() => {}} onSelectDatum={onSelectDatum} datumLabel={(id) => labels[id]} />);
    expect(screen.getByText(/Interpolated between:/)).toBeTruthy();
    expect(screen.queryByText("No sources for this value.")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Arm A, month 12" }));
    expect(onSelectDatum).toHaveBeenCalledWith("p6");
  });
});

describe("trial count prefers nct_ids_total", () => {
  it("shows the total when the server listed only the first ids", () => {
    render(<TrialList ids={["NCT00000001", "NCT00000002"]} total={57} />);
    const list = screen.getByRole("region", { name: "Trials" });
    expect(within(list).getByText("57 trials")).toBeTruthy();
    expect(within(list).getByText("Listing the first 2.")).toBeTruthy();
    expect(list.querySelectorAll("a").length).toBe(2);
  });

  it("falls back to the list length without a total", () => {
    render(<TrialList ids={["NCT00000001", "NCT00000002"]} />);
    expect(screen.getByText("2 trials")).toBeTruthy();
    expect(screen.queryByText(/Listing the first/)).toBeNull();
  });
});
