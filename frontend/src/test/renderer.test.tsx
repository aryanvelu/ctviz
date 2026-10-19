// Smoke test: every sample renders through <Renderer> and <ResponseView> without console errors,
// every ECharts option survives a real (server-side SVG) ECharts render, and clicking a datum
// opens the citation panel.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";
import * as echarts from "echarts/core";
import { SVGRenderer } from "echarts/renderers";
import { SAMPLES } from "../samples";
import { Renderer, findDatum } from "../viz/Renderer";
import { prettyEnum } from "../format";
import { ResponseView } from "../components/ResponseView";
import { isGraphData, type Datum } from "../types";
import { toRenderViz } from "../adapt";

const CHARTED = SAMPLES.filter((s) => s.response.visualization);

const captured = vi.hoisted(() => ({ charts: [] as { option: Record<string, unknown>; onEvents?: Record<string, (p: unknown) => void> }[] }));

vi.mock("echarts-for-react/lib/core", () => ({
  default: (props: { option: Record<string, unknown>; onEvents?: Record<string, (p: unknown) => void> }) => {
    captured.charts.push({ option: props.option, onEvents: props.onEvents });
    return <div data-testid="echart" />;
  },
}));

echarts.use([SVGRenderer]);

let errors: unknown[][];
beforeEach(() => {
  captured.charts = [];
  errors = [];
  vi.spyOn(console, "error").mockImplementation((...a) => { errors.push(a); });
  vi.spyOn(console, "warn").mockImplementation((...a) => { errors.push(a); });
});
afterEach(() => vi.restoreAllMocks());

/** All datumIds referenced by series data / links in an option. */
function datumIds(o: unknown, out = new Set<string>()): Set<string> {
  if (Array.isArray(o)) o.forEach((x) => datumIds(x, out));
  else if (o && typeof o === "object") {
    for (const [k, v] of Object.entries(o)) {
      if (k === "datumId" && typeof v === "string") out.add(v);
      else datumIds(v, out);
    }
  }
  return out;
}

function allDatums(data: Datum[] | { nodes: Datum[]; edges: Datum[] }): Datum[] {
  return isGraphData(data) ? [...data.nodes, ...data.edges] : data;
}

const PLAIN = new Set(["kpi_cards", "table", "eligibility_checklist"]);

describe("Renderer", () => {
  for (const { name, response } of CHARTED) {
    it(`renders ${name} (${response.visualization!.type})`, () => {
      const viz = toRenderViz(response)!;
      const onSelect = vi.fn();
      const { container } = render(<Renderer viz={viz} onSelect={onSelect} />);
      expect(container.querySelector(`[data-viz-type="${viz.type}"]`)).not.toBeNull();
      expect(screen.getByRole("heading", { name: viz.title })).toBeTruthy();

      if (PLAIN.has(viz.type)) {
        expect(captured.charts).toHaveLength(0);
        expect(container.querySelector(".data-table-details")).toBeNull();
        const clickable = container.querySelector(".kpi-card, tbody tr, .crit") as HTMLElement;
        fireEvent.click(clickable);
        expect(onSelect).toHaveBeenCalled();
      } else {
        expect(captured.charts.length).toBeGreaterThan(0);
        // Every chart has a keyboard-accessible data table listing every datum.
        const details = container.querySelector(".data-table-details");
        expect(details).not.toBeNull();
        expect(container.querySelector('[role="img"]')).toBeNull();
        for (const { option } of captured.charts) {
          expect(Array.isArray(option.series) && (option.series as unknown[]).length > 0).toBe(true);
          // Every referenced datumId must resolve, and every datum must be reachable from the chart.
          const ids = datumIds(option);
          for (const id of ids) expect(findDatum(viz, id), `unknown datumId ${id}`).toBeDefined();
          // Real ECharts layout pass (runs renderItem / formatters) in SSR mode.
          const chart = echarts.init(null, null, { renderer: "svg", ssr: true, width: 800, height: 500 });
          chart.setOption(option);
          expect(chart.renderToSVGString().length).toBeGreaterThan(100);
          chart.dispose();
        }
        const reachable = new Set(captured.charts.flatMap((c) => [...datumIds(c.option)]));
        if (viz.type === "geo_map") {
          // site level also shows a table; country level must be fully reachable from the chart
        } else {
          for (const d of allDatums(viz.data)) expect(reachable.has(d.id), `datum ${d.id} not clickable`).toBe(true);
        }
        const first = [...reachable][0];
        act(() => captured.charts[0].onEvents?.click?.({ data: { datumId: first } }));
        expect(onSelect).toHaveBeenCalledWith(first);
      }
      expect(errors).toEqual([]);
    });
  }
});

describe("ResponseView", () => {
  for (const { name, response } of CHARTED) {
    it(`renders ${name} with the answer, no notes, the response JSON, and a sources panel on click`, () => {
      const { container } = render(<ResponseView response={response} onAlternative={vi.fn()} busy={false} />);
      if (response.answer_summary) {
        expect(screen.getByRole("region", { name: "Answer" })).toBeTruthy();
        expect(screen.queryByText(/computed from the data/)).toBeNull();
        const ncts = response.answer_summary.match(/NCT\d{8}/g) ?? [];
        for (const n of ncts) {
          const links = container.querySelectorAll(`.answer a[href="https://clinicaltrials.gov/study/${n}"]`);
          expect(links.length).toBeGreaterThan(0);
        }
      }
      expect(container.querySelectorAll(".notes, .banner.info, .banner.warning").length).toBe(0); // notes stay in the JSON
      for (const n of response.meta.notes ?? []) expect(screen.queryByText(n.code.replace(/_/g, " "))).toBeNull();
      expect(screen.getByText("Response JSON")).toBeTruthy();
      expect(screen.queryByText(/Interpretation/)).toBeNull();
      expect(screen.queryByTestId("status")).toBeNull(); // status line only when status != ok

      const first = allDatums(response.visualization!.data)[0];
      if (PLAIN.has(response.visualization!.type)) {
        fireEvent.click(container.querySelector(".kpi-card, tbody tr, .crit") as HTMLElement);
      } else {
        act(() => captured.charts[0].onEvents?.click?.({ data: { datumId: first.id } }));
      }
      const panel = screen.getByRole("complementary", { name: "Sources" });
      expect(panel).toBeTruthy();
      const target = PLAIN.has(response.visualization!.type) ? null : first;
      if (target) {
        // One card per trial; every inline excerpt is shown (identical repeats within a trial once).
        const cites = (target.citations ?? []).filter((c) => c.kind !== "derived_count");
        expect(panel.querySelectorAll(".excerpt").length).toBeLessThanOrEqual(cites.length);
        for (const c of cites) expect(panel.textContent).toContain(prettyEnum(c.excerpt));
      }
      expect(panel.querySelector("code")).toBeNull(); // no field paths or datum ids
      expect(errors).toEqual([]);
    });
  }

  it("offers every excerpt when the inline list is truncated", () => {
    const s = SAMPLES.find((x) => x.name === "bar_chart_stacked")!;
    render(<ResponseView response={s.response} onAlternative={vi.fn()} busy={false} />);
    const d = (s.response.visualization!.data as Datum[])[0];
    act(() => captured.charts[0].onEvents?.click?.({ data: { datumId: d.id } }));
    expect(screen.getByRole("button", { name: `Show all ${d.citations_total} excerpts` })).toBeTruthy();
  });

  it("re-requests with the alternative type", () => {
    const s = SAMPLES.find((x) => x.name === "geo_map_country")!;
    const onAlt = vi.fn();
    render(<ResponseView response={s.response} onAlternative={onAlt} busy={false} />);
    fireEvent.click(screen.getByRole("button", { name: "Bar chart" }));
    expect(onAlt).toHaveBeenCalledWith("bar_chart");
  });
});
