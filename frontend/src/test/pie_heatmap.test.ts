// Pie slices are named by encoding.label (docs/response_schema.md §4.8), and a phase × status heatmap
// shows empty cells as 0 rather than "gap" (only the landscape gap heatmap asks for gap styling).
import { describe, expect, it } from "vitest";
import { heatmapOption, pieOption } from "../viz/options";
import { LIGHT } from "../theme";
import type { Visualization } from "../types";

const pie = (encoding: Visualization["encoding"]): Visualization => ({
  type: "pie_chart",
  title: "Trials by status: Alzheimer's disease",
  encoding,
  data: [
    { id: "d1", status: "Recruiting", trial_count: 614 },
    { id: "d2", status: "Completed", trial_count: 1900 },
  ],
  meta: { layout: { donut: true } },
} as unknown as Visualization);

const y = { field: "trial_count", type: "quantitative", title: "Trials" } as const;

describe("pie_chart", () => {
  it("draws slices from encoding.label (schema 2.x payload)", () => {
    const opt = pieOption(pie({ label: { field: "status", type: "nominal", title: "Status" }, y }), LIGHT);
    const data = (opt.series as { data: { name: string; value: number }[] }[])[0].data;
    expect(data.map((d) => [d.name, d.value])).toEqual([["Recruiting", 614], ["Completed", 1900]]);
  });

  it("still accepts encoding.x from older payloads", () => {
    const opt = pieOption(pie({ x: { field: "status", type: "nominal", title: "Status" }, y }), LIGHT);
    expect((opt.series as { data: unknown[] }[])[0].data).toHaveLength(2);
  });
});

describe("heatmap", () => {
  const heat = (highlight: boolean): Visualization => ({
    type: "heatmap",
    title: "Phase × status",
    encoding: {
      x: { field: "phase", type: "ordinal", title: "Phase" },
      y: { field: "status", type: "ordinal", title: "Status" },
      series: { field: "trial_count", type: "quantitative", title: "Trials" },
    },
    data: [
      { id: "d1", phase: "Phase 1", status: "Recruiting", trial_count: 3, is_gap: false },
      { id: "d2", phase: "Phase 2", status: "Recruiting", trial_count: 0, is_gap: true },
    ],
    meta: { highlight_zero: highlight },
  } as unknown as Visualization);

  const zeroCell = (v: Visualization) => {
    const s = (heatmapOption(v, LIGHT).series as { data: { datumId: string; itemStyle?: unknown }[]; label: { formatter: (p: unknown) => string } }[])[0];
    const cell = s.data.find((d) => d.datumId === "d2")!;
    return { styled: cell.itemStyle !== undefined, text: s.label.formatter({ data: cell, value: [1, 0, 0] }) };
  };

  it("shows a zero cell as 0 when gaps are not highlighted", () => {
    expect(zeroCell(heat(false))).toEqual({ styled: false, text: "0" });
  });

  it("marks a zero cell as a gap when highlight_zero is set", () => {
    expect(zeroCell(heat(true))).toEqual({ styled: true, text: "gap" });
  });

  it("never skips x-axis category labels", () => {
    const x = heatmapOption(heat(false), LIGHT).xAxis as { axisLabel: { interval: number } };
    expect(x.axisLabel.interval).toBe(0);
  });
});
