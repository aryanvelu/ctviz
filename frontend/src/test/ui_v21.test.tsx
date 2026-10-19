// Simplified UI (name, no example chips / debug, Enter submits) and the schema 2.1 citation shape:
// datum-level field_path, study links built from citation_policy.study_url, derived counts linking to
// datum.query_url, and the "of which planned" series on trial-start trends.
import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import App from "../App";
import { CitationPanel, studyLink } from "../components/CitationPanel";
import { LIGHT } from "../theme";
import { datumTooltip, timeSeriesOption } from "../viz/options";
import type { Datum, Visualization } from "../types";
import indexHtml from "../../index.html?raw";

vi.mock("echarts-for-react/lib/core", () => ({ default: () => <div data-testid="echart" /> }));

type Opt = Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any

afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); });

describe("simplified query form", () => {
  it("is called Clinical Trial Backend, with no example chips or debug checkbox", () => {
    const { container } = render(<App />);
    expect(screen.getByRole("heading", { level: 1 }).textContent).toBe("Clinical Trial Backend");
    expect(indexHtml).toContain("<title>Clinical Trial Backend</title>");
    expect(container.querySelector(".examples")).toBeNull();
    expect(screen.queryByLabelText("Include debug")).toBeNull();
    // kept: persona toggle, quick fields, more filters
    expect(screen.getByRole("radiogroup", { name: "Persona" })).toBeTruthy();
    expect(screen.getByLabelText("NCT ID")).toBeTruthy();
    expect(screen.getByRole("button", { name: /More filters/ })).toBeTruthy();
  });

  it("Enter submits, Shift+Enter does not", () => {
    const fn = vi.fn(() => new Promise<Response>(() => {}));
    vi.stubGlobal("fetch", fn);
    render(<App />);
    const q = screen.getByLabelText("Question");
    fireEvent.change(q, { target: { value: "semaglutide trials per year" } });
    fireEvent.keyDown(q, { key: "Enter", shiftKey: true });
    expect(fn).not.toHaveBeenCalled();
    fireEvent.keyDown(q, { key: "Enter" });
    expect(fn).toHaveBeenCalledTimes(1);
  });
});

describe("schema 2.1 citations", () => {
  const datum: Datum = {
    id: "d3", period: "2017-01-01", trial_count: 11, estimated_count: 0,
    nct_ids: ["NCT02692716", "NCT03010475"],
    field_path: "protocolSection.statusModule.startDateStruct.date",
    citations: [
      { nct_id: "NCT02692716", excerpt: "2017-01-17" },
      { nct_id: "NCT03010475", excerpt: "Adults with obesity", field_path: "protocolSection.eligibilityModule.eligibilityCriteria", kind: "text_excerpt", char_start: 0, char_end: 19 },
    ],
    citations_total: 11, excerpts_truncated: true,
  };

  it("shows only the trial link and the excerpt (no field paths, kinds, or offsets)", () => {
    const { container } = render(<CitationPanel datum={datum} title="2017" requestId="r" onClose={() => {}} />);
    const items = container.querySelectorAll("li.citation");
    expect(within(items[0] as HTMLElement).getByRole("link", { name: "NCT02692716" }).getAttribute("href")).toBe("https://clinicaltrials.gov/study/NCT02692716");
    expect(items[0].querySelector(".excerpt")!.textContent).toBe("2017-01-17");
    expect(items[1].querySelector(".excerpt")!.textContent).toBe("Adults with obesity");
    const text = container.textContent ?? "";
    for (const jargon of ["protocolSection", "field value", "text excerpt", "chars 0", "d3", "datum"]) expect(text).not.toContain(jargon);
    expect(container.querySelector("code, .tag")).toBeNull();
    expect(screen.getByText("2 trials")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Show all 11 excerpts" })).toBeTruthy();
  });

  it("groups one trial's excerpts in one card and drops identical repeats", () => {
    const d: Datum = {
      id: "e1", nct_ids: ["NCT01295827"],
      citations: [
        { nct_id: "NCT01295827", excerpt: "Biological: Pembrolizumab" },
        { nct_id: "NCT01295827", excerpt: "Biological: Pembrolizumab" },
        { nct_id: "NCT01295827", excerpt: "DRUG" },
      ],
      citations_total: 3364,
    };
    const { container } = render(<CitationPanel datum={d} title="Pembrolizumab" requestId="r" onClose={() => {}} />);
    expect(container.querySelectorAll("li.citation")).toHaveLength(1);
    expect([...container.querySelectorAll(".excerpt")].map((e) => e.textContent)).toEqual(["Biological: Pembrolizumab", "Drug"]);
    expect(screen.getByRole("button", { name: "Show all 3,364 excerpts" })).toBeTruthy();
  });

  it("builds study links from meta.citation_policy.study_url, refusing unsafe templates", () => {
    expect(studyLink("NCT02692716", "https://example.org/s/{nct_id}")).toBe("https://example.org/s/NCT02692716");
    expect(studyLink("NCT02692716", "javascript:{nct_id}")).toBe("https://clinicaltrials.gov/study/NCT02692716");
    expect(studyLink("not-an-id")).toBeUndefined();
    render(<CitationPanel datum={datum} title="2017" requestId="r" studyUrl="https://example.org/s/{nct_id}" onClose={() => {}} />);
    expect(screen.getAllByRole("link", { name: "NCT03010475" }).map((a) => a.getAttribute("href"))).toEqual(
      ["https://example.org/s/NCT03010475", "https://example.org/s/NCT03010475"], // trial list + citation
    );
  });

  it("a registry count (no nct_id, no url) links to datum.query_url in one plain line", () => {
    const d: Datum = {
      id: "d1", trial_count: 42, query_url: "https://clinicaltrials.gov/api/v2/studies?query.intr=semaglutide&countTotal=true",
      field_path: "$.totalCount", citations: [{ excerpt: "42", kind: "derived_count" }], citations_total: 1, excerpts_truncated: false,
    };
    const { container } = render(<CitationPanel datum={d} title="Phase 3" requestId="r" onClose={() => {}} />);
    expect(screen.getAllByRole("link")).toHaveLength(1);
    expect(screen.getByRole("link", { name: "open this search" }).getAttribute("href")).toBe(d.query_url);
    expect(container.textContent).not.toContain("totalCount");
  });

  it("accepts full citations from the all-citations endpoint (url + field_path per item)", async () => {
    const full = Array.from({ length: 11 }, (_, i) => ({
      nct_id: `NCT0000000${i % 10}`, excerpt: `e${i}`, field_path: "x.y", url: `https://clinicaltrials.gov/study/NCT0000000${i % 10}`,
    }));
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ citations: full }), { status: 200 })));
    const { container } = render(<CitationPanel datum={datum} title="2017" requestId="r" onClose={() => {}} />);
    fireEvent.click(screen.getByRole("button", { name: "Show all 11 excerpts" }));
    expect(await screen.findByText("e10")).toBeTruthy();
    expect(container.querySelectorAll("li.citation")).toHaveLength(11);
  });
});

describe("trial-start trends: of which planned", () => {
  const viz = (estimated: (number | null)[]): Visualization => ({
    type: "time_series", title: "Starts",
    encoding: {
      x: { field: "period", type: "temporal", title: "Start year", time_format: "%Y" },
      y: { field: "trial_count", type: "quantitative", title: "Trials" },
    },
    data: estimated.map((e, i) => ({
      id: `d${i}`, period: `${2024 + i}-01-01`, trial_count: 10 + i, estimated_count: e, citations: [{ nct_id: "NCT00000001", excerpt: "x" }],
    })),
  });

  it("adds a dashed second series when any estimated_count > 0, and 'planned: N' to the tooltip", () => {
    const v = viz([0, 2, 9]);
    const o = timeSeriesOption(v, LIGHT) as Opt;
    expect(o.series).toHaveLength(2);
    expect(o.series[1].name).toBe("of which planned (estimated start date)");
    expect(o.series[1].lineStyle.type).toBe("dashed");
    expect(o.series[1].data.map((p: Opt) => p.value)).toEqual([0, 2, 9]);
    expect(o.series[1].data.map((p: Opt) => p.datumId)).toEqual(["d0", "d1", "d2"]);
    expect(o.legend.show).toBe(true);
    expect(datumTooltip((v.data as Datum[])[2], v)).toContain("planned: <b>9</b>");
  });

  it("no second series when nothing is planned or the count was not computed", () => {
    for (const est of [[0, 0, 0], [null, null, null]]) {
      const v = viz(est);
      const o = timeSeriesOption(v, LIGHT) as Opt;
      expect(o.series).toHaveLength(1);
      expect(datumTooltip((v.data as Datum[])[0], v)).not.toContain("planned");
    }
  });
});
