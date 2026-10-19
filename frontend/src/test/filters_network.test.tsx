// Quick Status field (presets -> request.status), network edge weight labels, and fixed
// status_group series colors / stacking.
import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import App from "../App";
import { EMPTY_FORM, STATUS_PRESETS, statusesForPreset, statusPresetKey, toRequest } from "../components/QueryForm";
import { DARK, LIGHT, colorFor } from "../theme";
import { barOption, edgeStroke, networkOption, seriesColor, strongEdgeIndexes, timeSeriesOption } from "../viz/options";
import type { GraphData, Visualization } from "../types";

vi.mock("echarts-for-react/lib/core", () => ({ default: () => <div data-testid="echart" /> }));

type Opt = Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any

afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); });

describe("quick Status field", () => {
  it("maps each preset to the documented status list", () => {
    expect(statusesForPreset("")).toEqual([]);
    expect(statusesForPreset("active")).toEqual(["NOT_YET_RECRUITING", "RECRUITING", "ENROLLING_BY_INVITATION", "ACTIVE_NOT_RECRUITING"]);
    expect(statusesForPreset("recruiting")).toEqual(["RECRUITING"]);
    expect(statusesForPreset("done")).toEqual(["COMPLETED", "TERMINATED", "WITHDRAWN"]);
    expect(statusesForPreset("completed")).toEqual(["COMPLETED"]);
    expect(statusesForPreset("stopped")).toEqual(["TERMINATED", "WITHDRAWN"]);
    for (const p of STATUS_PRESETS) expect(p.label.length).toBeLessThanOrEqual(24);
  });

  it("recognises a preset regardless of order; anything else is custom", () => {
    expect(statusPresetKey([])).toBe("");
    expect(statusPresetKey(["WITHDRAWN", "COMPLETED", "TERMINATED"])).toBe("done");
    expect(statusPresetKey(["RECRUITING"])).toBe("recruiting");
    expect(statusPresetKey(["RECRUITING", "SUSPENDED"])).toBe("custom");
  });

  it("any status sends no status filter", () => {
    expect(toRequest({ ...EMPTY_FORM, query: "q" }).status).toEqual([]);
    expect(toRequest({ ...EMPTY_FORM, query: "q", status: statusesForPreset("done") }).status).toEqual(["COMPLETED", "TERMINATED", "WITHDRAWN"]);
  });

  it("the select writes the same status array the More filters chips show", async () => {
    const fn = vi.fn((_url: string, _init?: RequestInit) => new Promise<Response>(() => {}));
    vi.stubGlobal("fetch", fn);
    render(<App />);
    const sel = screen.getByLabelText("Status") as HTMLSelectElement;
    expect(sel.value).toBe("");
    fireEvent.change(sel, { target: { value: "active" } });
    expect(sel.value).toBe("active");
    expect(sel.title).toMatch(/recruiting, not yet recruiting/);
    fireEvent.click(screen.getByRole("button", { name: /More filters/ }));
    const chip = (name: RegExp) => (screen.getByRole("checkbox", { name }) as HTMLInputElement);
    expect(chip(/^Recruiting$/).checked).toBe(true);
    expect(chip(/^Completed$/).checked).toBe(false);
    // Hand-picking a chip turns the quick field into "custom".
    fireEvent.click(chip(/^Completed$/));
    expect(sel.value).toBe("custom");
    fireEvent.change(sel, { target: { value: "" } });
    expect(chip(/^Recruiting$/).checked).toBe(false);
    fireEvent.change(sel, { target: { value: "stopped" } });
    fireEvent.change(screen.getByLabelText("Question"), { target: { value: "trials" } });
    fireEvent.click(screen.getByRole("button", { name: "Visualize" }));
    const body = JSON.parse(String(fn.mock.calls[0][1]?.body));
    expect(body.status).toEqual(["TERMINATED", "WITHDRAWN"]);
  });
});

function graph(weights: number[], nodeType: (i: number) => string = () => "drug"): Visualization {
  const ids = Array.from({ length: weights.length + 1 }, (_, i) => `n${i}`);
  const data: GraphData = {
    nodes: ids.map((id, i) => ({ id, label: `Node ${i}`, type: nodeType(i), citations: [] })),
    edges: weights.map((w, i) => ({ id: `e${i}`, source: "n0", target: ids[i + 1], weight: w, citations: [] })),
  };
  return { type: "network_graph", title: "g", encoding: { nodes: { id: "id", label: "label", type: "type" }, edges: { source: "source", target: "target", weight: "weight" } }, data, meta: { network_kind: "drug_drug" } } as Visualization;
}

describe("network edge labels", () => {
  it("labels every weighted edge in a small graph", () => {
    expect([...strongEdgeIndexes([3, null, 1])].sort()).toEqual([0, 2]);
  });

  it("labels only the strongest edges (>= 80th percentile, at most 15) in a big graph", () => {
    const w = Array.from({ length: 40 }, (_, i) => i + 1); // 1..40 -> p80 = 32
    const idx = strongEdgeIndexes(w);
    expect(idx.size).toBe(9);
    for (const i of idx) expect(w[i]).toBeGreaterThanOrEqual(32);
    const many = Array.from({ length: 200 }, (_, i) => i);
    expect(strongEdgeIndexes(many).size).toBe(15);
    expect(strongEdgeIndexes(many).has(199)).toBe(true);
  });

  it("width 1-8 px and opacity grow with sqrt(weight)", () => {
    expect(edgeStroke(100, 100)).toEqual({ width: 8, opacity: 0.85 });
    expect(edgeStroke(0, 100)).toEqual({ width: 1, opacity: 0.2 });
    expect(edgeStroke(25, 100).width).toBe(4.5);
    expect(edgeStroke(null, 100).width).toBe(1);
  });

  it("puts a weight label on the strong links only, and the tooltip says 'A + B: N trials'", () => {
    const viz = graph(Array.from({ length: 30 }, (_, i) => i + 1));
    const o = networkOption(viz, LIGHT) as Opt;
    const links = o.series[0].links as Opt[];
    const shown = links.filter((l) => l.label?.show);
    expect(shown.length).toBe(7); // 1..30 -> p80 = 24
    expect(shown.map((l) => l.label.formatter)).toContain("30");
    expect(links[0].label).toBeUndefined();
    expect(links[29].lineStyle.width).toBeGreaterThan(links[0].lineStyle.width);
    expect(links[29].lineStyle.opacity).toBeGreaterThan(links[0].lineStyle.opacity);
    const html = o.tooltip.formatter({ dataType: "edge", data: { datumId: "e29" } });
    expect(html).toContain("Node 0 + Node 30</strong>: <b>30 trials</b>");
  });

  it("other network kinds read as 'A – B: N trials'", () => {
    const viz = { ...graph([5, 2], (i) => (i === 0 ? "sponsor" : "drug")), meta: { network_kind: "sponsor_drug" } };
    const o = networkOption(viz, LIGHT) as Opt;
    expect(o.tooltip.formatter({ dataType: "edge", data: { datumId: "e0" } })).toContain("Node 0 – Node 1</strong>: <b>5 trials</b>");
  });
});

const GROUPS = ["Active", "Done (completed, terminated, withdrawn)", "Other (suspended, unknown, expanded access)"];

function statusRows() {
  return ["2020", "2021"].flatMap((year) => GROUPS.map((g, i) => ({ id: `${year}-${i}`, year, status_group: g, n: i + 1, citations: [] })));
}

describe("status_group series", () => {
  it("gets fixed colors (active green, done slate, other amber) in both themes", () => {
    const sg = { field: "status_group", type: "nominal", title: "Status" } as const;
    expect(seriesColor(LIGHT, sg, GROUPS[0], 0)).toBe("#2f9e44");
    expect(seriesColor(LIGHT, sg, GROUPS[1], 1)).toBe("#64748b");
    expect(seriesColor(LIGHT, sg, GROUPS[2], 2)).toBe("#d9a02b");
    expect(seriesColor(DARK, sg, GROUPS[0], 0)).toBe("#6ed37f");
    // Other series fields keep the palette.
    expect(seriesColor(LIGHT, { ...sg, field: "phase" }, "Active", 1)).toBe(colorFor(LIGHT, 1));
  });

  it("a stacked bar chart stacks and colors by status group", () => {
    const viz = {
      type: "bar_chart", title: "t", data: statusRows(),
      encoding: { x: { field: "year", type: "ordinal", title: "Year" }, y: { field: "n", type: "quantitative", title: "Trials" }, series: { field: "status_group", type: "nominal", title: "Status" } },
      meta: { layout: { mode: "stacked" } },
    } as unknown as Visualization;
    const o = barOption(viz, LIGHT) as Opt;
    expect(o.series.map((s: Opt) => s.stack)).toEqual(["total", "total", "total"]);
    expect(o.series.map((s: Opt) => s.itemStyle.color)).toEqual(["#2f9e44", "#64748b", "#d9a02b"]);
  });

  it("a time series colors by status group and stacks when layout.mode is stacked", () => {
    const base = {
      type: "time_series", title: "t", data: statusRows(),
      encoding: { x: { field: "year", type: "temporal", title: "Year" }, y: { field: "n", type: "quantitative", title: "Trials" }, series: { field: "status_group", type: "nominal", title: "Status" } },
    } as unknown as Visualization;
    const plain = timeSeriesOption(base, LIGHT) as Opt;
    expect(plain.series.map((s: Opt) => s.itemStyle.color)).toEqual(["#2f9e44", "#64748b", "#d9a02b"]);
    expect(plain.series[0].stack).toBeUndefined();
    const stacked = timeSeriesOption({ ...base, meta: { layout: { mode: "stacked" } } } as Visualization, LIGHT) as Opt;
    expect(stacked.series.every((s: Opt) => s.stack === "total" && s.areaStyle)).toBe(true);
  });
});

describe("dense force graphs", () => {
  it("label at most one edge per node, so labels spread instead of stacking on hubs", () => {
    // Hub h0 has the 3 strongest edges; only its strongest is labelled, then disjoint pairs.
    const ends: [string, string][] = [["h0", "a"], ["h0", "b"], ["h0", "c"], ["x", "y"], ["a", "z"]];
    expect([...strongEdgeIndexes([50, 49, 48, 10, 9], 10, ends, 1)].sort()).toEqual([0, 3]);
  });

  it("a graph with more than 3 edges per node uses the one-label-per-node rule", () => {
    const n = 6;
    const nodes = Array.from({ length: n }, (_, i) => ({ id: `n${i}`, label: `N${i}`, type: "drug", citations: [] }));
    const edges = [];
    let k = 0;
    for (let a = 0; a < n; a++) for (let b = a + 1; b < n; b++) edges.push({ id: `e${k}`, source: `n${a}`, target: `n${b}`, weight: 100 - k++, citations: [] });
    // 15 edges vs 6 nodes is not dense; duplicate the pairs to cross 3x.
    for (let a = 0; a < n; a++) for (let b = a + 1; b < n; b++) edges.push({ id: `e${k}`, source: `n${a}`, target: `n${b}`, weight: 100 - k++, citations: [] });
    const viz = { type: "network_graph", title: "g", encoding: { nodes: { id: "id", label: "label", type: "type" }, edges: { source: "source", target: "target", weight: "weight" } }, data: { nodes, edges } } as unknown as Visualization;
    const links = (networkOption(viz, LIGHT) as Opt).series[0].links as Opt[];
    const shown = links.filter((l) => l.label?.show);
    const touched = shown.flatMap((l) => [l.source, l.target]);
    expect(shown.length).toBeGreaterThan(0);
    expect(new Set(touched).size).toBe(touched.length);
  });
});
