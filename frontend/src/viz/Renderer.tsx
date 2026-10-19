// The one generic renderer: dispatches on visualization.type to a component per type.
import { Component, useMemo, type ComponentType, type ErrorInfo, type ReactNode } from "react";
import type { Datum, Visualization, VizType } from "../types";
import { isGraphData } from "../types";
import { formatValue, humanize, prettyEnum } from "../format";
import { useChartTheme, type ChartTheme } from "../theme";
import { EChart, type EOption } from "./EChart";
import {
  barOption, forestOption, geoCountryFields, geoCountryOption, geoSiteOption, heatmapOption, histogramOption,
  networkOption, pieOption, sankeyOption, scatterOption, survivalOption, thresholdOption, timelineHeight,
  timelineOption, timeSeriesOption,
} from "./options";
import { ChartDataTable, Checklist, DataTable, KpiCards, scalarEntries } from "./Plain";

export interface RendererProps {
  viz: Visualization;
  onSelect: (datumId: string) => void;
  selectedId?: string;
}

type OptionBuilder = (viz: Visualization, theme: ChartTheme) => EOption;

function rowCount(viz: Visualization, field?: string): number {
  const rows = Array.isArray(viz.data) ? viz.data : [];
  return field ? new Set(rows.map((r) => String(r[field]))).size : rows.length;
}

function chartComponent(build: OptionBuilder, height: (viz: Visualization) => number = () => 420): ComponentType<RendererProps> {
  return function ChartView({ viz, onSelect }: RendererProps) {
    const theme = useChartTheme();
    // Built once per (response, theme): clicking a datum or any other re-render must not hand ECharts
    // a new option, which would redraw (and re-animate) the chart from scratch.
    const option = useMemo(() => build(viz, theme), [viz, theme]);
    return <EChart option={option} height={height(viz)} onSelect={onSelect} ariaLabel={viz.title} />;
  };
}

function GeoMap(props: RendererProps) {
  const theme = useChartTheme();
  const { viz, onSelect } = props;
  const geo = viz.meta?.geo;
  const site = geo?.level === "site" && !!geo.lat_field && !!geo.lon_field;
  const country = !site && geo?.level === "country" && !!geoCountryFields(viz);
  const option = useMemo(
    () => (site ? geoSiteOption(viz, theme) : country ? geoCountryOption(viz, theme) : null),
    [viz, theme, site, country],
  );
  if (site && option) {
    return <EChart option={option} height={360} onSelect={onSelect} ariaLabel={`${viz.title} (site map)`} />;
  }
  if (country && option) {
    return <EChart option={option} height={Math.max(260, Math.min(1200, rowCount(viz) * 30 + 100))} onSelect={onSelect} ariaLabel={viz.title} />;
  }
  return <p className="banner warning">This map cannot be drawn; the data table below has every value.</p>;
}

const horizontalBarHeight = (viz: Visualization) =>
  (viz.meta?.layout?.orientation === "horizontal" ? Math.max(260, Math.min(1200, rowCount(viz, viz.encoding.x?.field) * 40 + 110)) : 420);

export const RENDERERS: Record<VizType, ComponentType<RendererProps>> = {
  kpi_cards: KpiCards,
  table: DataTable,
  bar_chart: chartComponent(barOption, horizontalBarHeight),
  histogram: chartComponent(histogramOption),
  time_series: chartComponent(timeSeriesOption),
  scatter_plot: chartComponent(scatterOption),
  heatmap: chartComponent(heatmapOption, () => 460),
  pie_chart: chartComponent(pieOption),
  network_graph: chartComponent(networkOption, () => 560),
  sankey: chartComponent(sankeyOption, () => 460),
  timeline: chartComponent(timelineOption, timelineHeight),
  survival_curve: chartComponent(survivalOption),
  threshold_line: chartComponent(thresholdOption),
  power_curve: chartComponent(thresholdOption),
  forest_plot: chartComponent(forestOption, (v) => Math.max(260, Math.min(1200, rowCount(v) * 36 + 100))),
  geo_map: GeoMap,
  eligibility_checklist: Checklist,
};

/** Types drawn as plain React (they are their own accessible table). */
const PLAIN = new Set<VizType>(["kpi_cards", "table", "eligibility_checklist"]);
const GRAPH_TYPES = new Set<VizType>(["network_graph", "sankey"]);

const THRESHOLD_PREFIX = "meta.thresholds[";

/** Find a datum (row, node, edge, or a cited threshold line) by id. */
export function findDatum(viz: Visualization, id: string): Datum | undefined {
  if (id.startsWith(THRESHOLD_PREFIX)) {
    const t = viz.meta?.thresholds?.[Number(id.slice(THRESHOLD_PREFIX.length, -1))];
    return t ? { id, label: t.label, citations: t.citations ?? [], ...(t.citations?.length ? {} : { cite_note: `Source: ${t.source ?? "convention"}` }) } : undefined;
  }
  if (isGraphData(viz.data)) return viz.data.nodes.find((n) => n.id === id) ?? viz.data.edges.find((e) => e.id === id);
  return viz.data.find((d) => d.id === id);
}

/** Catches a rendering crash and shows the raw visualization JSON instead of a blank page. */
export class VizErrorBoundary extends Component<{ viz: Visualization; children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidUpdate(prev: { viz: Visualization }) {
    if (prev.viz !== this.props.viz && this.state.error) this.setState({ error: null });
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Keep the details available to developers without breaking the page.
    console.info("visualization render failed", error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="banner error" role="alert">
        <strong>This {humanize(this.props.viz.type)} could not be drawn.</strong> {this.state.error.message}
        <details open>
          <summary>Raw visualization JSON</summary>
          <pre className="raw-json">{JSON.stringify(this.props.viz, null, 2)}</pre>
        </details>
      </div>
    );
  }
}

function KeyValues({ title, entries }: { title: string; entries: [string, unknown][] }) {
  if (!entries.length) return null;
  return (
    <div className="viz-kv">
      <p className="eyebrow">{title}</p>
      <dl className="viz-summary">
        {entries.map(([k, v]) => <div key={k}><dt>{humanize(k)}</dt><dd>{typeof v === "string" ? prettyEnum(v) : formatValue(v)}</dd></div>)}
      </dl>
    </div>
  );
}

function Thresholds({ viz, onSelect }: Pick<RendererProps, "viz" | "onSelect">) {
  const ts = viz.meta?.thresholds ?? [];
  if (!ts.length) return null;
  return (
    <ul className="thresholds" aria-label="Reference lines">
      {ts.map((t, i) => (
        <li key={i}>
          <span className="threshold-swatch" aria-hidden="true" />
          {t.label} <span className="muted small">({formatValue(t.value)})</span>
          {!!t.citations?.length && (
            <button type="button" className="cite-btn" onClick={() => onSelect(`${THRESHOLD_PREFIX}${i}]`)}>Sources</button>
          )}
        </li>
      ))}
    </ul>
  );
}

export function Renderer({ viz, onSelect, selectedId }: RendererProps) {
  const Comp = RENDERERS[viz.type];
  const graph = isGraphData(viz.data);
  if (!Comp || GRAPH_TYPES.has(viz.type) !== graph) {
    return (
      <div className="banner warning" role="alert">
        {!Comp ? <>Unsupported visualization type “{viz.type}”.</> : <>Malformed response: {viz.type} expects {GRAPH_TYPES.has(viz.type) ? "{nodes, edges}" : "a list of rows"}.</>}
        <pre className="raw-json">{JSON.stringify(viz, null, 2)}</pre>
      </div>
    );
  }
  const empty = graph ? (viz.data as { nodes: Datum[] }).nodes.length === 0 : (viz.data as Datum[]).length === 0;
  const approx = viz.meta?.approximation;
  const plain = PLAIN.has(viz.type);
  return (
    <figure className="viz" data-viz-type={viz.type}>
      <figcaption>
        <h2>{viz.title}</h2>
        {viz.subtitle && <p className="subtitle">{viz.subtitle}</p>}
        {approx?.is_approximation && (
          <div className="viz-badges">
            <details className="approx">
              <summary><span className="badge approx-badge">Approximation</span> {approx.method}</summary>
              {!!approx.assumptions?.length && <ul>{approx.assumptions.map((a) => <li key={a}>{a}</li>)}</ul>}
            </details>
          </div>
        )}
      </figcaption>
      <VizErrorBoundary viz={viz}>
        {empty ? <p className="muted">No data to display.</p> : <Comp viz={viz} onSelect={onSelect} selectedId={selectedId} />}
        <Thresholds viz={viz} onSelect={onSelect} />
        {!plain && !empty && (
          <details className="data-table-details" open={viz.type === "geo_map" && viz.meta?.geo?.level === "site"}>
            <summary>Data table</summary>
            <ChartDataTable viz={viz} onSelect={onSelect} selectedId={selectedId} />
          </details>
        )}
      </VizErrorBoundary>
      {viz.type !== "eligibility_checklist" && <KeyValues title="Summary" entries={scalarEntries(viz.meta?.summary)} />}
      <KeyValues title="Statistics" entries={scalarEntries(viz.meta?.stats)} />
      {Object.keys(viz.meta?.diagnostics ?? {}).length > 0 && (
        <details className="diagnostics">
          <summary className="small">How this chart was computed</summary>
          <KeyValues title="Diagnostics" entries={scalarEntries(viz.meta?.diagnostics)} />
        </details>
      )}
    </figure>
  );
}
