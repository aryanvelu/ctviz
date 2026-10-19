// Pure functions: Visualization spec -> ECharts option. Channel field names always come from
// `visualization.encoding` / `visualization.meta`; the only data keys read directly are the reserved
// ones documented in docs/response_schema_conventions.md (see RESERVED_KEYS in ../types).
// Missing numbers are never defaulted to 0: the datum is skipped (and still listed in the data table).
import type { Channel, Datum, GraphData, TimelineLane, Visualization, VizMeta } from "../types";
import {
  axisFormatter, axisName, categories, escapeHtml, formatChannel, formatValue, num, prettyEnum, str,
  tooltipLines,
} from "../format";
import { axisStyle, baseOption, colorFor, type ChartTheme } from "../theme";
import type { EOption } from "./EChart";

/* eslint-disable @typescript-eslint/no-explicit-any */

const rowsOf = (viz: Visualization): Datum[] => (Array.isArray(viz.data) ? viz.data : []);

/** Tooltip HTML for one datum: title line, encoded channel values, extra tooltip fields, citation hint. */
export function datumTooltip(d: Datum | undefined, viz: Visualization, title?: string): string {
  if (!d) return "";
  const enc = viz.encoding;
  const meta = viz.meta;
  const lines: string[] = [];
  if (title) lines.push(`<strong>${escapeHtml(title)}</strong>`);
  const chans: (Channel | undefined)[] = [enc.label, enc.x, enc.x2, enc.y, enc.y2, enc.series, enc.size, enc.error_low, enc.error_high];
  const seen = new Set<string>();
  for (const c of chans) {
    if (!c || seen.has(c.field) || d[c.field] === undefined) continue;
    seen.add(c.field);
    lines.push(`${escapeHtml(c.title)}: <b>${escapeHtml(prettyMaybe(formatChannel(d, c, meta), c))}</b>`);
  }
  for (const l of tooltipLines(d, enc.tooltip?.filter((f) => !seen.has(f)), meta, enc)) lines.push(escapeHtml(l));
  const planned = plannedCount(d);
  if (viz.type === "time_series" && planned !== null && hasPlanned(rowsOf(viz))) lines.push(`planned: <b>${escapeHtml(formatValue(planned))}</b>`);
  lines.push(citeHint(d));
  return lines.filter(Boolean).join("<br/>");
}

function citeHint(d: Datum): string {
  const n = d.citations_total ?? d.citations?.length ?? 0;
  if (!n && d.cite_note) return `<i>${escapeHtml(d.cite_note)}</i>`;
  return n ? `<span style="opacity:.75">Click to see the sources</span>` : "";
}

function prettyMaybe(s: string, c: Channel): string {
  return c.type === "nominal" || c.type === "ordinal" ? prettyEnum(s) : s;
}

function byId(rows: Datum[]): Map<string, Datum> {
  return new Map(rows.map((r) => [r.id, r]));
}

/** Themed tooltip merged with chart-specific settings. */
export function tip(theme: ChartTheme, extra: EOption): EOption {
  return { ...baseOption(theme).tooltip, ...extra };
}

function itemTooltip(viz: Visualization, lookup: Map<string, Datum>, theme: ChartTheme, title?: (d: Datum) => string) {
  return tip(theme, {
    trigger: "item",
    formatter: (p: any) => {
      const d = lookup.get(p?.data?.datumId);
      return d ? datumTooltip(d, viz, title?.(d)) : escapeHtml(String(p?.name ?? ""));
    },
  });
}

/* ------------------------------------------------------------ log scales */

/** A channel asks for a log scale but some values are <= 0, so the chart falls back to linear. */
export function logFallbacks(viz: Visualization): string[] {
  const rows = rowsOf(viz);
  const enc = viz.encoding;
  const out: string[] = [];
  const check = (ch: Channel | undefined, extra: (Channel | undefined)[] = [], more: number[] = []) => {
    if (ch?.scale?.type !== "log") return;
    const fields = [ch, ...extra].filter((c): c is Channel => !!c).map((c) => c.field);
    if (!allPositive(rows, fields, more)) out.push(ch.title);
  };
  const ref = viz.type === "forest_plot" && viz.meta?.reference_line ? [viz.meta.reference_line.value] : [];
  check(enc.x, viz.type === "forest_plot" ? [enc.error_low, enc.error_high] : [], ref);
  check(enc.y, viz.type === "forest_plot" ? [] : [enc.error_low, enc.error_high]);
  return out;
}

function allPositive(rows: Datum[], fields: string[], more: number[] = []): boolean {
  for (const v of more) if (!(v > 0)) return false;
  for (const r of rows) {
    for (const f of fields) {
      const v = num(r[f]);
      if (v !== null && v <= 0) return false;
    }
  }
  return true;
}

function valueAxis(theme: ChartTheme, ch: Channel | undefined, meta: VizMeta | undefined, extra: EOption = {}, logOk = true): EOption {
  const log = ch?.scale?.type === "log" && logOk;
  const domain = ch?.scale?.domain;
  const fmt = axisFormatter(ch);
  return {
    type: log ? "log" : "value",
    name: axisName(ch, meta),
    nameLocation: "middle",
    nameGap: 36,
    scale: ch?.scale?.zero === false,
    ...(domain && typeof domain[0] === "number" && typeof domain[1] === "number" ? { min: domain[0], max: domain[1] } : {}),
    ...axisStyle(theme),
    axisLabel: { color: theme.muted, ...(fmt ? { formatter: fmt } : {}) },
    ...extra,
  };
}

function categoryAxis(theme: ChartTheme, ch: Channel | undefined, meta: VizMeta | undefined, data: string[], extra: EOption = {}): EOption {
  const fmt = axisFormatter(ch);
  return {
    type: "category",
    data,
    name: axisName(ch, meta),
    nameLocation: "middle",
    nameGap: 36,
    ...axisStyle(theme),
    axisLabel: {
      color: theme.muted,
      hideOverlap: true,
      formatter: (v: string) => truncate(prettyEnum(fmt ? fmt(v) : v), 28),
    },
    splitLine: { show: false },
    ...extra,
  };
}

/** A truncation "Other" bucket always goes last, whatever the sort. */
function otherLast(cats: string[], meta: VizMeta | undefined): string[] {
  if (!meta?.truncation?.other_bucket) return cats;
  const i = cats.findIndex((c) => c.toLowerCase() === "other");
  return i < 0 ? cats : [...cats.slice(0, i), ...cats.slice(i + 1), cats[i]];
}

function truncate(s: string, n: number): string {
  return s.length > n ? `${s.slice(0, n - 1)}…` : s;
}

/* ------------------------------------------------------ status groups */
// Series split by `status_group` ("Active" / "Done (...)" / "Other (...)") get fixed meaning-bearing
// colors (active = green, done = slate, other = amber); every other series keeps the palette order.
const STATUS_GROUP_COLORS: Record<string, { light: string; dark: string }> = {
  active: { light: "#2f9e44", dark: "#6ed37f" },
  done: { light: "#64748b", dark: "#94a3b8" },
  other: { light: "#d9a02b", dark: "#f2b84b" },
};

export function statusGroupColor(theme: ChartTheme, key: string): string | null {
  const k = key.trim().toLowerCase();
  const hit = Object.keys(STATUS_GROUP_COLORS).find((g) => k === g || k.startsWith(`${g} `) || k.startsWith(`${g}(`));
  return hit ? STATUS_GROUP_COLORS[hit][theme.dark ? "dark" : "light"] : null;
}

/** Color for series `si` with value `key`; status_group series use the fixed status colors. */
export function seriesColor(theme: ChartTheme, series: Channel | undefined, key: string, si: number): string {
  return (series?.field === "status_group" ? statusGroupColor(theme, key) : null) ?? colorFor(theme, si);
}

const GRID = { left: 56, right: 24, top: 48, bottom: 56, containLabel: true };

/* ------------------------------------------------------------------ bar */

export function barOption(viz: Visualization, theme: ChartTheme): EOption {
  const rows = rowsOf(viz);
  const meta = viz.meta;
  // Convention: x is always the category axis; orientation only decides how it is drawn.
  const { x: cat, y: val, series } = viz.encoding;
  if (!cat || !val) return {};
  const layout = meta?.layout ?? {};
  const horizontal = layout.orientation === "horizontal";
  const mode = layout.mode ?? "grouped";
  const cats = otherLast(categories(rows, cat.field, cat.sort ?? meta?.sort, val.field), meta);
  const keys = series ? categories(rows, series.field, series.sort) : [""];
  const lookup = byId(rows);
  const totals = new Map<string, number>();
  if (mode === "normalized") {
    for (const r of rows) {
      const v = num(r[val.field]);
      if (v !== null) totals.set(str(r[cat.field]), (totals.get(str(r[cat.field])) ?? 0) + v);
    }
  }
  const errLo = viz.encoding.error_low;
  const errHi = viz.encoding.error_high;
  const k = keys.length;
  const logOk = allPositive(rows, [val.field, errLo?.field, errHi?.field].filter((f): f is string => !!f));

  const out: EOption[] = keys.map((key, si) => {
    const data = cats.map((c) => {
      const r = rows.find((row) => str(row[cat.field]) === c && (!series || str(row[series.field]) === key));
      if (!r) return { value: null };
      let v = num(r[val.field]);
      if (mode === "normalized" && v !== null) {
        const t = totals.get(c);
        v = t ? v / t : null;
      }
      return { value: v, datumId: r.id, ...(r.cite_note && !r.citations?.length ? { itemStyle: { opacity: 0.6 } } : {}) };
    });
    return {
      type: "bar",
      name: series ? prettyEnum(key) : val.title,
      data,
      itemStyle: { color: seriesColor(theme, series, key, si) },
      barCategoryGap: "20%",
      barGap: "30%",
      ...(mode !== "grouped" ? { stack: "total" } : {}),
      emphasis: { focus: "series" },
    };
  });

  if (errLo && errHi && mode === "grouped") {
    keys.forEach((key, si) => {
      const data = cats.flatMap((c, ci) => {
        const r = rows.find((row) => str(row[cat.field]) === c && (!series || str(row[series.field]) === key));
        const lo = r ? num(r[errLo.field]) : null;
        const hi = r ? num(r[errHi.field]) : null;
        return r && lo !== null && hi !== null ? [{ value: [ci, lo, hi], datumId: r.id }] : [];
      });
      out.push({
        type: "custom",
        name: series ? prettyEnum(key) : val.title,
        data,
        z: 10,
        tooltip: { show: false },
        renderItem: (_params: any, api: any) => {
          const ci = api.value(0);
          const band = api.size(horizontal ? [0, 1] : [1, 0])[horizontal ? 1 : 0];
          const usable = band * 0.8;
          const bw = usable / (k + (k - 1) * 0.3);
          const offset = -usable / 2 + si * bw * 1.3 + bw / 2;
          const p1 = api.coord(horizontal ? [api.value(1), ci] : [ci, api.value(1)]);
          const p2 = api.coord(horizontal ? [api.value(2), ci] : [ci, api.value(2)]);
          const cap = Math.min(8, bw / 3);
          const style = { stroke: theme.text, lineWidth: 1.5 };
          const shape = horizontal
            ? [[p1[0], p1[1] + offset, p2[0], p2[1] + offset], [p1[0], p1[1] + offset - cap, p1[0], p1[1] + offset + cap], [p2[0], p2[1] + offset - cap, p2[0], p2[1] + offset + cap]]
            : [[p1[0] + offset, p1[1], p2[0] + offset, p2[1]], [p1[0] + offset - cap, p1[1], p1[0] + offset + cap, p1[1]], [p2[0] + offset - cap, p2[1], p2[0] + offset + cap, p2[1]]];
          return { type: "group", children: shape.map(([x1, y1, x2, y2]) => ({ type: "line", shape: { x1, y1, x2, y2 }, style })) };
        },
      });
    });
  }

  const valFmt = mode === "normalized" ? (v: number) => formatValue(v, ".0%") : axisFormatter(val);
  const vAxis = valueAxis(theme, val, meta, mode === "normalized" ? { max: 1, axisLabel: { color: theme.muted, formatter: valFmt } } : {}, logOk && mode !== "stacked");
  const cAxis = categoryAxis(theme, cat, meta, cats, horizontal ? { inverse: true, name: "" } : {});
  return {
    ...baseOption(theme),
    legend: { ...baseOption(theme).legend, show: !!series },
    tooltip: itemTooltip(viz, lookup, theme),
    grid: { ...GRID, ...(horizontal ? { left: 16 } : {}) },
    xAxis: horizontal ? vAxis : { ...cAxis, nameGap: cats.length > 8 ? 64 : 36, axisLabel: { ...cAxis.axisLabel, rotate: cats.length > 8 ? 30 : 0 } },
    yAxis: horizontal ? { ...cAxis, axisLabel: { ...cAxis.axisLabel, width: 180, overflow: "truncate" } } : vAxis,
    series: out,
  };
}

/* ------------------------------------------------------------ histogram */

/** Bins with a start and a count. A bin without an end (open-ended) is drawn one typical width wide. */
export function histogramBins(viz: Visualization): { d: Datum; start: number; end: number; count: number; open: boolean }[] {
  const { x, x2, y } = viz.encoding;
  if (!x || !y) return [];
  const raw = rowsOf(viz).map((d) => ({
    d, start: num(d[x.field]), end: x2 ? num(d[x2.field]) : null, count: num(d[y.field]),
  })).filter((b): b is { d: Datum; start: number; end: number | null; count: number } => b.start !== null && b.count !== null);
  const widths = raw.filter((b) => b.end !== null && b.end > b.start).map((b) => b.end! - b.start).sort((a, b) => a - b);
  const typical = widths.length ? widths[Math.floor(widths.length / 2)] : 1;
  return raw.map((b) => ({ d: b.d, start: b.start, end: b.end ?? b.start + typical, count: b.count, open: b.end === null }));
}

export function histogramOption(viz: Visualization, theme: ChartTheme): EOption {
  const { x, x2, y } = viz.encoding;
  if (!x || !y) return {};
  const lookup = byId(rowsOf(viz));
  const bins = histogramBins(viz);
  const data = bins.map((b) => ({ value: [b.start, b.end, b.count, b.open ? 1 : 0], datumId: b.d.id }));
  return {
    ...baseOption(theme),
    legend: { show: false },
    tooltip: itemTooltip(viz, lookup, theme, (d) => {
      const s = formatValue(d[x.field], x.format);
      const e = x2 ? d[x2.field] : undefined;
      return e === undefined || e === null ? `≥ ${s} (open-ended)` : `${s} – ${formatValue(e, x2?.format ?? x.format)}`;
    }),
    grid: GRID,
    xAxis: valueAxis(theme, x, viz.meta, { scale: true }),
    yAxis: valueAxis(theme, y, viz.meta),
    series: [{
      type: "custom",
      name: y.title,
      data,
      encode: { x: [0, 1], y: 2 },
      renderItem: (_p: any, api: any) => {
        const a = api.coord([api.value(0), api.value(2)]);
        const b = api.coord([api.value(1), 0]);
        const open = api.value(3) === 1;
        return {
          type: "rect",
          shape: { x: a[0] + 1, y: a[1], width: Math.max(1, b[0] - a[0] - 2), height: b[1] - a[1] },
          style: open
            ? { fill: colorFor(theme, 0), opacity: 0.45, stroke: colorFor(theme, 0), lineDash: [4, 3] }
            : { fill: colorFor(theme, 0) },
        };
      },
    }],
  };
}

/* ---------------------------------------------------------- time_series */

function sortTemporal(values: string[]): string[] {
  return [...values].sort((a, b) => {
    const na = Number(a);
    const nb = Number(b);
    if (Number.isFinite(na) && Number.isFinite(nb)) return na - nb;
    return a.localeCompare(b);
  });
}

/** Trial-start trends: how many of a point's trials have an estimated (planned) start date; null = not computed. */
function plannedCount(d: Datum): number | null {
  return typeof d.estimated_count === "number" && Number.isFinite(d.estimated_count) ? d.estimated_count : null;
}

export function hasPlanned(rows: Datum[]): boolean {
  return rows.some((r) => (plannedCount(r) ?? 0) > 0);
}

export function timeSeriesOption(viz: Visualization, theme: ChartTheme): EOption {
  const rows = rowsOf(viz);
  const { x, y, series } = viz.encoding;
  if (!x || !y) return {};
  const lookup = byId(rows);
  const present = rows.filter((r) => r[x.field] !== undefined && r[x.field] !== null);
  const xs = x.sort?.values?.length ? categories(present, x.field, x.sort) : sortTemporal([...new Set(present.map((r) => str(r[x.field])))]);
  const keys = series ? categories(rows, series.field, series.sort) : [""];
  const stacked = !!series && viz.meta?.layout?.mode === "stacked";
  return {
    ...baseOption(theme),
    legend: { ...baseOption(theme).legend, show: !!series || hasPlanned(present) },
    tooltip: itemTooltip(viz, lookup, theme),
    grid: GRID,
    xAxis: categoryAxis(theme, x, viz.meta, xs, { boundaryGap: false }),
    yAxis: valueAxis(theme, y, viz.meta, {}, !stacked && allPositive(rows, [y.field])),
    dataZoom: xs.length > 40 ? [{ type: "inside" }, { type: "slider", height: 18, bottom: 8 }] : [],
    series: keys.flatMap((key, si) => {
      const at = xs.map((xv) => present.find((row) => str(row[x.field]) === xv && (!series || str(row[series.field]) === key)));
      const line = {
        type: "line",
        name: series ? prettyEnum(key) : y.title,
        showSymbol: true,
        symbolSize: 7,
        connectNulls: false,
        itemStyle: { color: seriesColor(theme, series, key, si) },
        lineStyle: { width: 2 },
        // Stacked: each band's height is that group's count, the top line is the total.
        ...(stacked ? { stack: "total", areaStyle: { opacity: 0.35 } } : {}),
        emphasis: { focus: "series" },
        data: at.map((r) => {
          const v = r ? num(r[y.field]) : null;
          return r && v !== null ? { value: v, datumId: r.id } : { value: null };
        }),
      };
      if (!hasPlanned(at.filter((r): r is Datum => !!r))) return [line];
      // Second, lighter dashed line: the part of each count whose start date is still an estimate.
      const planned = {
        type: "line",
        name: series ? `${prettyEnum(key)}: of which planned` : "of which planned (estimated start date)",
        showSymbol: true,
        symbol: "emptyCircle",
        symbolSize: 6,
        connectNulls: false,
        itemStyle: { color: seriesColor(theme, series, key, si), opacity: 0.6 },
        lineStyle: { width: 1.5, type: "dashed", opacity: 0.6 },
        emphasis: { focus: "series" },
        data: at.map((r) => {
          const v = r ? plannedCount(r) : null;
          return r && v !== null ? { value: v, datumId: r.id } : { value: null };
        }),
      };
      return [line, planned];
    }),
  };
}

/* --------------------------------------------------------- scatter_plot */

export function toTime(v: unknown): number | null {
  if (typeof v === "number") return Number.isFinite(v) ? v : null;
  const s = str(v);
  if (!s) return null;
  if (/^\d{4}$/.test(s)) return Date.UTC(Number(s), 0, 1);
  if (/^\d{4}-\d{2}$/.test(s)) return Date.parse(`${s}-01T00:00:00Z`);
  const t = Date.parse(s);
  return Number.isFinite(t) ? t : null;
}

function sizeScaler(rows: Datum[], field: string | undefined, min = 8, max = 40): (d: Datum) => number {
  if (!field) return () => 10;
  const vals = rows.map((r) => num(r[field])).filter((v): v is number => v !== null && v >= 0);
  const hi = Math.sqrt(Math.max(1, ...vals));
  return (d) => {
    const v = num(d[field]);
    return v === null ? min : min + ((max - min) * Math.sqrt(Math.max(0, v))) / hi;
  };
}

export function scatterOption(viz: Visualization, theme: ChartTheme): EOption {
  const rows = rowsOf(viz);
  const { x, y, series, size, label } = viz.encoding;
  if (!x || !y) return {};
  const lookup = byId(rows);
  const keys = series ? categories(rows, series.field, series.sort) : [""];
  const scale = sizeScaler(rows, size?.field);
  const xTemporal = x.type === "temporal";
  const xCat = x.type === "nominal" || x.type === "ordinal";
  const xCats = xCat ? categories(rows, x.field, x.sort) : [];
  const xVal = (r: Datum) => (xTemporal ? toTime(r[x.field]) : xCat ? (r[x.field] === undefined ? null : str(r[x.field])) : num(r[x.field]));
  return {
    ...baseOption(theme),
    legend: { ...baseOption(theme).legend, show: !!series },
    tooltip: itemTooltip(viz, lookup, theme, label ? (d) => str(d[label.field]) : undefined),
    grid: GRID,
    xAxis: xCat ? categoryAxis(theme, x, viz.meta, xCats)
      : xTemporal ? { ...valueAxis(theme, x, viz.meta), type: "time", axisLabel: { color: theme.muted } }
        : valueAxis(theme, x, viz.meta, { scale: x.scale?.zero !== true }, allPositive(rows, [x.field])),
    yAxis: valueAxis(theme, y, viz.meta, {}, allPositive(rows, [y.field])),
    series: keys.map((key, si) => ({
      type: "scatter",
      name: series ? prettyEnum(key) : y.title,
      itemStyle: { color: colorFor(theme, si), opacity: 0.85, borderColor: theme.bg, borderWidth: 1 },
      emphasis: { focus: "series", label: { show: true } },
      label: label ? { show: rows.length <= 20, position: "top", color: theme.text, fontSize: 11, formatter: (p: any) => str(lookup.get(p.data.datumId)?.[label.field]) } : undefined,
      data: rows
        .filter((r) => !series || str(r[series.field]) === key)
        .map((r) => ({ xv: xVal(r), yv: num(r[y.field]), r }))
        .filter((p) => p.xv !== null && p.yv !== null)
        .map(({ xv, yv, r }) => ({ value: [xv, yv], symbolSize: scale(r), datumId: r.id })),
    })),
  };
}

/* -------------------------------------------------------------- heatmap */

export function heatmapOption(viz: Visualization, theme: ChartTheme): EOption {
  const rows = rowsOf(viz);
  const { x, y, series } = viz.encoding;
  if (!x || !y || !series) return {};
  const lookup = byId(rows);
  const xs = categories(rows, x.field, x.sort);
  const ys = categories(rows, y.field, y.sort);
  const vals = rows.map((r) => num(r[series.field])).filter((v): v is number => v !== null);
  const fmt = axisFormatter(series);
  const highlight = viz.meta?.highlight_zero === true;
  return {
    ...baseOption(theme),
    legend: { show: false },
    tooltip: itemTooltip(viz, lookup, theme, (d) => (highlight && d.is_gap === true ? "Gap: no trials" : "")),
    grid: { ...GRID, right: 96 },
    // Every column is a category the reader must be able to name, so never skip x labels; tilt them instead.
    xAxis: (() => {
      const axis = categoryAxis(theme, x, viz.meta, xs, { splitArea: { show: false } });
      return { ...axis, nameGap: 56, axisLabel: { ...axis.axisLabel, interval: 0, hideOverlap: false, rotate: xs.length > 6 ? 30 : 0 } };
    })(),
    yAxis: { ...categoryAxis(theme, y, viz.meta, ys), inverse: true, name: "", axisLabel: { color: theme.muted, width: 160, overflow: "truncate", formatter: (v: string) => prettyEnum(v) } },
    visualMap: {
      min: Math.min(0, ...vals),
      max: Math.max(1, ...vals),
      calculable: true,
      orient: "vertical",
      right: 4,
      top: "middle",
      itemHeight: 160,
      text: [axisName(series, viz.meta), ""],
      textStyle: { color: theme.muted },
      inRange: { color: theme.sequential },
      ...(fmt ? { formatter: (v: number) => fmt(v) } : {}),
    },
    series: [{
      type: "heatmap",
      name: series.title,
      data: rows
        .map((r) => ({ r, v: num(r[series.field]) }))
        .filter(({ v }) => v !== null)
        .map(({ r, v }) => ({
          value: [xs.indexOf(str(r[x.field])), ys.indexOf(str(r[y.field])), v],
          datumId: r.id,
          ...(highlight && r.is_gap === true ? { itemStyle: { borderColor: theme.gap, borderWidth: 2, borderType: "dashed" } } : {}),
        })),
      label: {
        show: rows.length <= 120,
        color: theme.text,
        textBorderColor: theme.bg,
        textBorderWidth: 2,
        formatter: (p: any) => (highlight && lookup.get(p.data.datumId)?.is_gap === true ? "gap" : fmt ? fmt(p.value[2]) : formatValue(p.value[2])),
      },
      itemStyle: { borderColor: theme.bg, borderWidth: 1 },
      emphasis: { itemStyle: { borderColor: theme.text, borderWidth: 2 } },
    }],
  };
}

/* ------------------------------------------------------------ pie_chart */

export function pieOption(viz: Visualization, theme: ChartTheme): EOption {
  const rows = rowsOf(viz);
  // The contract names the slice category `label` (docs/response_schema.md §4.8); `x` is accepted from older payloads.
  const cat = viz.encoding.label ?? viz.encoding.x;
  const val = viz.encoding.y;
  if (!cat || !val) return {};
  const lookup = byId(rows);
  const donut = viz.meta?.layout?.donut;
  const cats = otherLast(categories(rows, cat.field, cat.sort ?? viz.meta?.sort, val.field), viz.meta);
  return {
    ...baseOption(theme),
    legend: { ...baseOption(theme).legend, bottom: 0, top: undefined, formatter: (n: string) => prettyEnum(n) },
    tooltip: itemTooltip(viz, lookup, theme),
    series: [{
      type: "pie",
      name: val.title,
      radius: donut ? ["42%", "68%"] : ["0%", "68%"],
      center: ["50%", "46%"],
      avoidLabelOverlap: true,
      itemStyle: { borderColor: theme.bg, borderWidth: 2 },
      label: { color: theme.text, formatter: (p: any) => `${truncate(prettyEnum(p.name), 22)}\n${p.percent}%` },
      data: cats.flatMap((c, i) => {
        const r = rows.find((row) => str(row[cat.field]) === c);
        const v = r ? num(r[val.field]) : null;
        return r && v !== null ? [{ name: c, value: v, datumId: r.id, itemStyle: { color: colorFor(theme, i) } }] : [];
      }),
    }],
  };
}

/* -------------------------------------------------------- network_graph */

function stripNull<T extends object>(o: T | undefined): Partial<T> {
  if (!o) return {};
  return Object.fromEntries(Object.entries(o).filter(([, v]) => v !== null && v !== undefined)) as Partial<T>;
}

/** Node/edge field names: encoding.nodes / encoding.edges with the schema defaults. */
export function graphFields(viz: Visualization) {
  return {
    ne: { id: "id", label: "label", type: "type", size: "size", ...stripNull(viz.encoding.nodes) },
    ee: { source: "source", target: "target", weight: "weight", type: "type", ...stripNull(viz.encoding.edges) },
  };
}

/**
 * Ego layout: pinned node(s) at the center, every other node on concentric rings, strongest link to
 * the center first (ring 1 holds 10 nodes, each further ring 8 more). Deterministic, so a re-render
 * never reshuffles the picture.
 */
export function egoPositions(g: GraphData, ee: { source: string; target: string; weight?: string | null }, pinned: Set<string>): Map<string, { x: number; y: number }> {
  const pos = new Map<string, { x: number; y: number }>();
  const centers = g.nodes.filter((n) => pinned.has(n.id));
  centers.forEach((n, i) => {
    const a = (2 * Math.PI * i) / Math.max(1, centers.length);
    const r = centers.length > 1 ? 40 : 0;
    pos.set(n.id, { x: r * Math.cos(a), y: r * Math.sin(a) });
  });
  const strength = new Map<string, number>();
  for (const e of g.edges) {
    const s = str(e[ee.source]);
    const t = str(e[ee.target]);
    const w = (ee.weight ? num(e[ee.weight]) : null) ?? 1;
    if (pinned.has(s) && !pinned.has(t)) strength.set(t, (strength.get(t) ?? 0) + w);
    if (pinned.has(t) && !pinned.has(s)) strength.set(s, (strength.get(s) ?? 0) + w);
  }
  const others = g.nodes.filter((n) => !pinned.has(n.id))
    .sort((a, b) => (strength.get(b.id) ?? 0) - (strength.get(a.id) ?? 0) || a.id.localeCompare(b.id));
  let i = 0;
  for (let ring = 1; i < others.length; ring++) {
    const cap = 10 + 8 * (ring - 1);
    const members = others.slice(i, i + cap);
    members.forEach((n, k) => {
      const a = (2 * Math.PI * k) / members.length + (ring % 2 ? 0 : Math.PI / members.length);
      pos.set(n.id, { x: 160 * ring * Math.cos(a), y: 160 * ring * Math.sin(a) });
    });
    i += cap;
  }
  return pos;
}

/** Above this many nodes the force refinement is skipped (spiral start positions only). */
export const FORCE_MAX_NODES = 500;

/**
 * Deterministic force-directed positions (Fruchterman-Reingold with a fixed iteration count).
 * The layout is computed once here and drawn with ECharts `layout: "none"`, so the picture never
 * jiggles, re-rendering cannot restart a simulation, and the same graph always looks the same.
 * Nodes start on a sunflower spiral ordered by degree (hubs in the middle, ties by id); there is no
 * randomness. Heavier edges pull their ends closer. Above FORCE_MAX_NODES nodes only the spiral is
 * used. Returns positions in arbitrary units (ECharts
 * fits them into the chart area).
 */
export function forcePositions(
  ids: string[],
  links: { source: string; target: string; weight?: number | null }[],
  opts: { iterations?: number } = {},
): Map<string, { x: number; y: number }> {
  const n = ids.length;
  const pos = new Map<string, { x: number; y: number }>();
  if (!n) return pos;
  const index = new Map(ids.map((id, i) => [id, i]));
  const edges = links.flatMap((l) => {
    const a = index.get(l.source);
    const b = index.get(l.target);
    return a === undefined || b === undefined || a === b ? [] : [{ a, b, w: l.weight ?? 1 }];
  });
  const degree = new Array<number>(n).fill(0);
  for (const lk of edges) { degree[lk.a] += 1; degree[lk.b] += 1; }
  const wMax = Math.max(1, ...edges.map((lk) => lk.w));
  const order = ids.map((_, i) => i).sort((a, b) => degree[b] - degree[a] || (ids[a] < ids[b] ? -1 : ids[a] > ids[b] ? 1 : a - b));
  const k = 100; // ideal edge length
  const xs = new Float64Array(n);
  const ys = new Float64Array(n);
  const golden = Math.PI * (3 - Math.sqrt(5));
  order.forEach((node, rank) => {
    const r = k * 0.6 * Math.sqrt(rank + 0.5);
    xs[node] = r * Math.cos(rank * golden);
    ys[node] = r * Math.sin(rank * golden);
  });
  // The O(n^2) refinement is skipped for huge graphs so drawing never blocks the main thread;
  // the degree-ordered spiral alone still puts the hubs in the middle.
  const iterations = n > FORCE_MAX_NODES ? 0 : opts.iterations ?? Math.max(60, Math.min(300, Math.round(60000 / n)));
  const dx = new Float64Array(n);
  const dy = new Float64Array(n);
  let temp = k * Math.sqrt(n) * 0.5;
  const cool = temp / (iterations + 1);
  for (let it = 0; it < iterations; it++) {
    dx.fill(0);
    dy.fill(0);
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        let vx = xs[i] - xs[j];
        let vy = ys[i] - ys[j];
        let d2 = vx * vx + vy * vy;
        if (d2 < 1e-6) { // coincident: push apart along a fixed, index-derived direction
          vx = Math.cos(i + j);
          vy = Math.sin(i + j);
          d2 = 1;
        }
        const f = (k * k) / d2; // = (k^2 / d) / d, applied to the unnormalized vector
        dx[i] += vx * f; dy[i] += vy * f;
        dx[j] -= vx * f; dy[j] -= vy * f;
      }
    }
    for (const lk of edges) {
      const vx = xs[lk.a] - xs[lk.b];
      const vy = ys[lk.a] - ys[lk.b];
      const d = Math.sqrt(vx * vx + vy * vy) || 1e-3;
      const f = (d / k) * (0.5 + 0.5 * Math.min(1, lk.w / wMax)); // = (d^2 / k) / d
      dx[lk.a] -= vx * f; dy[lk.a] -= vy * f;
      dx[lk.b] += vx * f; dy[lk.b] += vy * f;
    }
    for (let i = 0; i < n; i++) {
      // Mild gravity keeps disconnected pieces from drifting away.
      dx[i] -= xs[i] * 0.02;
      dy[i] -= ys[i] * 0.02;
      const len = Math.sqrt(dx[i] * dx[i] + dy[i] * dy[i]);
      if (len > 0) {
        const step = Math.min(len, temp);
        xs[i] += (dx[i] / len) * step;
        ys[i] += (dy[i] / len) * step;
      }
    }
    temp = Math.max(temp - cool, k * 0.01);
  }
  ids.forEach((id, i) => pos.set(id, { x: Math.round(xs[i] * 100) / 100, y: Math.round(ys[i] * 100) / 100 }));
  return pos;
}

/**
 * Which edges get a weight label: every weighted edge when there are at most `max`, otherwise the
 * strongest `max` among those at or above the 80th-percentile weight (ties keep data order).
 * With `ends` + `perNode`, a node carries at most `perNode` labelled edges, so labels on hub-heavy
 * graphs spread out instead of stacking around the hubs.
 */
export function strongEdgeIndexes(weights: (number | null)[], max = 15, ends?: [string, string][], perNode = Infinity): Set<number> {
  const ranked = weights.flatMap((w, i) => (w === null ? [] : [{ w, i }])).sort((a, b) => b.w - a.w || a.i - b.i);
  let pool = ranked;
  if (ranked.length > max) {
    const asc = ranked.map((x) => x.w).sort((a, b) => a - b);
    const p80 = asc[Math.min(asc.length - 1, Math.ceil(0.8 * asc.length) - 1)];
    pool = ranked.filter((x) => x.w >= p80);
  }
  const used = new Map<string, number>();
  const out = new Set<number>();
  for (const x of pool) {
    if (out.size >= max) break;
    const pair = ends?.[x.i];
    if (pair && pair.some((node) => (used.get(node) ?? 0) >= perNode)) continue;
    if (pair) for (const node of pair) used.set(node, (used.get(node) ?? 0) + 1);
    out.add(x.i);
  }
  return out;
}

/** Edge width (1-8 px) and opacity (0.2-0.85), both on a sqrt scale of weight / max weight. */
export function edgeStroke(w: number | null, wMax: number): { width: number; opacity: number } {
  if (w === null || wMax <= 0) return { width: 1, opacity: 0.45 };
  const t = Math.sqrt(Math.max(0, w) / wMax);
  return { width: Math.round((1 + 7 * t) * 10) / 10, opacity: Math.round((0.2 + 0.65 * t) * 100) / 100 };
}

export function networkOption(viz: Visualization, theme: ChartTheme): EOption {
  const g = viz.data as GraphData;
  const { ne, ee } = graphFields(viz);
  const nodeLookup = byId(g.nodes);
  const edgeLookup = byId(g.edges);
  const declared = viz.meta?.node_types ?? [];
  const typeKeys = [...declared.map((t) => t.type)];
  for (const n of g.nodes) {
    const t = ne.type ? str(n[ne.type]) : "";
    if (t && !typeKeys.includes(t)) typeKeys.push(t);
  }
  if (!typeKeys.length) typeKeys.push("");
  const typeLabel = (t: string) => declared.find((nt) => nt.type === t)?.label ?? (t ? prettyEnum(t) : "Node");
  const pinned = new Set(viz.meta?.pinned ?? []);
  const sizeOf = sizeScaler(g.nodes, ne.size ?? undefined, 10, 48);
  const weights = g.edges.map((e) => (ee.weight ? num(e[ee.weight]) : null));
  const wMax = Math.max(1, ...weights.filter((w): w is number => w !== null));
  const big = g.nodes.length > 60;
  const ego = viz.meta?.network_layout === "ego" ? egoPositions(g, ee, pinned) : null;
  // Positions are computed once, deterministically; ECharts never runs a live force simulation.
  const positions = ego ?? forcePositions(
    g.nodes.map((n) => n.id),
    g.edges.map((e, i) => ({ source: str(e[ee.source]), target: str(e[ee.target]), weight: weights[i] })),
  );
  // Dense force layouts put the strongest (hub-to-hub) edges side by side, so label fewer of them;
  // hovering a node still shows the weights of all its edges.
  const edgeWeight = new Map(g.edges.map((e, i) => [e.id, weights[i]]));
  const dense = !ego && g.edges.length > 3 * g.nodes.length;
  const ends = g.edges.map((e): [string, string] => [str(e[ee.source]), str(e[ee.target])]);
  const labelled = dense ? strongEdgeIndexes(weights, 10, ends, 1) : strongEdgeIndexes(weights);
  return {
    ...baseOption(theme),
    legend: { ...baseOption(theme).legend, data: typeKeys.map(typeLabel) },
    tooltip: tip(theme, {
      trigger: "item",
      formatter: (p: any) => {
        if (p.dataType === "edge") {
          const e = edgeLookup.get(p.data.datumId);
          if (!e) return "";
          const s = nodeLookup.get(str(e[ee.source]));
          const t = nodeLookup.get(str(e[ee.target]));
          // drug_drug pairs read as a combination ("A + B"); every other kind as a link ("A – B").
          const joiner = viz.meta?.directed ? "→" : viz.meta?.network_kind === "drug_drug" ? "+" : "–";
          const pair = `${str(s?.[ne.label] ?? e[ee.source])} ${joiner} ${str(t?.[ne.label] ?? e[ee.target])}`;
          const w = ee.weight ? e[ee.weight] : undefined;
          const unit = (ee.weight && viz.meta?.units?.[ee.weight]) || "trials";
          const lines = [w !== undefined && w !== null
            ? `<strong>${escapeHtml(pair)}</strong>: <b>${escapeHtml(formatValue(w))} ${escapeHtml(unit)}</b>`
            : `<strong>${escapeHtml(pair)}</strong>`];
          for (const l of tooltipLines(e, viz.encoding.tooltip, viz.meta, viz.encoding)) lines.push(escapeHtml(l));
          lines.push(citeHint(e));
          return lines.filter(Boolean).join("<br/>");
        }
        const d = nodeLookup.get(p.data?.datumId);
        if (!d) return "";
        const lines = [`<strong>${escapeHtml(str(d[ne.label]))}</strong>`];
        if (ne.type && d[ne.type] !== undefined) lines.push(escapeHtml(typeLabel(str(d[ne.type]))));
        if (ne.size && d[ne.size] !== undefined) {
          const u = viz.meta?.units?.[ne.size];
          lines.push(`${escapeHtml(prettyEnum(ne.size))}: <b>${escapeHtml(formatValue(d[ne.size]))}${u ? ` ${escapeHtml(u)}` : ""}</b>`);
        }
        if (pinned.has(d.id)) lines.push("<i>Named in your question</i>");
        lines.push(citeHint(d));
        return lines.filter(Boolean).join("<br/>");
      },
    }),
    series: [{
      type: "graph",
      // 'ego': every node links to the pinned node, so it is drawn centered with the others on rings around it.
      // Fixed positions (ego rings or the precomputed force layout): nothing animates after the
      // first draw, and dragging a node moves only that node.
      layout: "none",
      roam: true,
      draggable: true,
      top: 40,
      categories: typeKeys.map((t, i) => ({ name: typeLabel(t), itemStyle: { color: colorFor(theme, i) } })),
      label: { show: !big, position: "right", color: theme.text, fontSize: 11, formatter: (p: any) => truncate(str(nodeLookup.get(p.data.datumId)?.[ne.label]), 26) },
      labelLayout: { hideOverlap: true },
      emphasis: {
        focus: "adjacency", lineStyle: { width: 6 },
        edgeLabel: { show: true, fontSize: 10, color: theme.text, backgroundColor: theme.bg, borderColor: theme.grid, borderWidth: 1, borderRadius: 3, padding: [1, 3], formatter: (p: any) => {
          const w = edgeWeight.get(p.data?.datumId);
          return w === null || w === undefined ? "" : formatValue(w);
        } },
      },
      lineStyle: { color: theme.muted, opacity: 0.55, curveness: 0.08 },
      edgeLabel: { show: false, fontSize: 10 },
      ...(viz.meta?.directed ? { edgeSymbol: ["none", "arrow"], edgeSymbolSize: 8 } : {}),
      data: g.nodes.map((n) => {
        const ti = Math.max(0, typeKeys.indexOf(ne.type ? str(n[ne.type]) : ""));
        return {
          id: n.id, name: n.id, category: ti, symbolSize: sizeOf(n), datumId: n.id,
          ...positions.get(n.id),
          ...(pinned.has(n.id) ? {
            itemStyle: { borderColor: theme.text, borderWidth: 3 },
            // Ego center: label above the node on a background, clear of the radiating edges' weight labels.
            label: { show: true, fontWeight: "bold", ...(ego ? { position: "top", backgroundColor: theme.bg, padding: [1, 4], borderRadius: 3 } : {}) },
          } : {}),
        };
      }),
      links: g.edges.map((e, i) => ({
        source: str(e[ee.source]),
        target: str(e[ee.target]),
        datumId: e.id,
        lineStyle: edgeStroke(weights[i], wMax),
        ...(labelled.has(i) ? {
          label: {
            show: true, formatter: formatValue(weights[i]), fontSize: 10, color: theme.text,
            backgroundColor: theme.bg, borderColor: theme.grid, borderWidth: 1, borderRadius: 3, padding: [1, 3],
          },
        } : {}),
      })),
    }],
  };
}

/* --------------------------------------------------------------- sankey */

export function sankeyOption(viz: Visualization, theme: ChartTheme): EOption {
  const g = viz.data as GraphData;
  const { ne, ee } = graphFields(viz);
  const nodeLookup = byId(g.nodes);
  const edgeLookup = byId(g.edges);
  const stages = viz.meta?.stages ?? [];
  const unit = ee.weight ? viz.meta?.units?.[ee.weight] ?? "" : "";
  return {
    ...baseOption(theme),
    legend: { show: false },
    tooltip: tip(theme, {
      trigger: "item",
      formatter: (p: any) => {
        if (p.dataType === "edge") {
          const e = edgeLookup.get(p.data.datumId);
          if (!e) return "";
          const s = str(nodeLookup.get(str(e[ee.source]))?.[ne.label]);
          const t = str(nodeLookup.get(str(e[ee.target]))?.[ne.label]);
          const w = ee.weight ? formatValue(e[ee.weight]) : "";
          return [`<strong>${escapeHtml(s)} → ${escapeHtml(t)}</strong>`, escapeHtml(`${w} ${unit}`.trim()), citeHint(e)].filter(Boolean).join("<br/>");
        }
        const d = nodeLookup.get(p.data?.datumId);
        if (!d) return "";
        const stage = typeof d.stage === "number" ? stages[d.stage] : undefined;
        return [`<strong>${escapeHtml(str(d[ne.label]))}</strong>`, stage ? escapeHtml(stage) : "", escapeHtml(`${formatValue(p.value)} ${unit}`.trim()), citeHint(d)].filter(Boolean).join("<br/>");
      },
    }),
    series: [{
      type: "sankey",
      top: 24,
      bottom: 24,
      left: 8,
      right: 140,
      nodeGap: 12,
      emphasis: { focus: "adjacency" },
      lineStyle: { color: "gradient", opacity: 0.35 },
      label: { color: theme.text, formatter: (p: any) => truncate(str(nodeLookup.get(p.data.datumId)?.[ne.label]), 30) },
      data: g.nodes.map((n, i) => ({
        name: n.id,
        datumId: n.id,
        ...(typeof n.stage === "number" ? { depth: n.stage } : {}),
        itemStyle: { color: colorFor(theme, i) },
      })),
      links: g.edges.flatMap((e) => {
        const w = ee.weight ? num(e[ee.weight]) : null;
        return w === null || w <= 0 ? [] : [{ source: str(e[ee.source]), target: str(e[ee.target]), value: w, datumId: e.id }];
      }),
    }],
  };
}

/* ------------------------------------------------------------- timeline */

export interface TimelineGroup {
  kind: "calendar" | "relative";
  unit?: string;
  origin?: string;
  lanes: TimelineLane[];
  rows: Datum[];
  labels: string[];
}

/**
 * Lanes from meta.lanes (rows join a lane by their encoding.series value); without meta.lanes, one
 * lane built from encoding.x / encoding.x2 and meta.time_axis. Lanes are grouped by axis kind
 * (and unit), and each group gets its own x axis.
 */
export function timelineGroups(viz: Visualization): TimelineGroup[] {
  const rows = rowsOf(viz);
  const { x, x2, y, series } = viz.encoding;
  let lanes: TimelineLane[] = viz.meta?.lanes ?? [];
  if (!lanes.length) {
    if (!x) return [];
    const kind = viz.meta?.time_axis?.kind ?? (x.type === "temporal" ? "calendar" : "relative");
    lanes = [{ name: "", axis: { kind, unit: viz.meta?.time_axis?.unit, origin: viz.meta?.time_axis?.origin }, start_field: x.field, end_field: x2?.field }];
  }
  const laneOf = (r: Datum): TimelineLane | undefined => {
    if (lanes.length === 1 && !lanes[0].name) return lanes[0];
    if (series) return lanes.find((l) => l.name === str(r[series.field]));
    return lanes.find((l) => r[l.start_field] !== undefined);
  };
  const groups: TimelineGroup[] = [];
  for (const lane of lanes) {
    const key = `${lane.axis.kind}|${lane.axis.kind === "relative" ? lane.axis.unit ?? "" : ""}`;
    let g = groups.find((x) => `${x.kind}|${x.kind === "relative" ? x.unit ?? "" : ""}` === key);
    if (!g) {
      g = { kind: lane.axis.kind, unit: lane.axis.unit, origin: lane.axis.origin, lanes: [], rows: [], labels: [] };
      groups.push(g);
    }
    g.lanes.push(lane);
  }
  for (const r of rows) {
    const lane = laneOf(r);
    if (!lane) continue;
    const g = groups.find((x) => x.lanes.includes(lane))!;
    g.rows.push(r);
  }
  for (const g of groups) {
    g.labels = y ? categories(g.rows, y.field, y.sort) : g.rows.map((r) => r.id);
  }
  return groups.filter((g) => g.rows.length);
}

const LANE_ROW_PX = 40;
const GROUP_GAP_PX = 64;

/** Pixel height for the timeline (one block per axis group). */
export function timelineHeight(viz: Visualization): number {
  const groups = timelineGroups(viz);
  const rows = groups.reduce((a, g) => a + g.labels.length, 0);
  return Math.max(240, Math.min(1600, rows * LANE_ROW_PX + groups.length * GROUP_GAP_PX + 60));
}

export function timelineOption(viz: Visualization, theme: ChartTheme): EOption {
  const { y, series, label } = viz.encoding;
  const lookup = byId(rowsOf(viz));
  const groups = timelineGroups(viz);
  if (!groups.length) return {};
  const allLanes = groups.flatMap((g) => g.lanes);
  const laneColor = (lane: TimelineLane) => colorFor(theme, allLanes.indexOf(lane));
  const grids: EOption[] = [];
  const xAxes: EOption[] = [];
  const yAxes: EOption[] = [];
  const out: EOption[] = [];
  let top = 40;
  groups.forEach((g, gi) => {
    const h = g.labels.length * LANE_ROW_PX;
    grids.push({ left: 16, right: 32, top, height: h, containLabel: true });
    top += h + GROUP_GAP_PX;
    const tv = (v: unknown) => (g.kind === "calendar" ? toTime(v) : num(v));
    const unitWord = g.unit ?? "";
    // Pad the extent so point events at the ends are not clipped.
    const ext = g.rows.flatMap((r) => g.lanes.flatMap((l) => [tv(r[l.start_field]), l.end_field ? tv(r[l.end_field]) : null]))
      .filter((v): v is number => v !== null);
    const lo = ext.length ? Math.min(...ext) : 0;
    const hi = ext.length ? Math.max(...ext) : 1;
    const padX = (hi - lo) * 0.05 || (g.kind === "calendar" ? 30 * 864e5 : 1);
    const bounds = { min: lo - padX, max: hi + padX };
    xAxes.push(g.kind === "calendar"
      ? { type: "time", gridIndex: gi, name: "Date", nameLocation: "middle", nameGap: 28, ...bounds, ...axisStyle(theme) }
      : {
        type: "value", gridIndex: gi, nameLocation: "middle", nameGap: 28, ...bounds, ...axisStyle(theme),
        axisLabel: { color: theme.muted, formatter: (v: number) => (Number.isInteger(v) ? formatValue(v) : "") },
        name: `${unitWord ? humanizeUnit(unitWord) : "Offset"}${g.origin ? ` from ${g.origin}` : ""}`,
      });
    yAxes.push({ ...categoryAxis(theme, y, viz.meta, g.labels), gridIndex: gi, inverse: true, name: "", axisLabel: { color: theme.muted, width: 200, overflow: "truncate" } });
    g.lanes.forEach((lane, li) => {
      const color = laneColor(lane);
      const data = g.rows
        .filter((r) => (lane.name && series ? str(r[series.field]) === lane.name : true))
        .flatMap((r) => {
          const s = tv(r[lane.start_field]);
          if (s === null) return [];
          const rawEnd = lane.end_field ? r[lane.end_field] : undefined;
          const e = rawEnd === undefined || rawEnd === null ? null : tv(rawEnd);
          return [{ value: [g.labels.indexOf(y ? str(r[y.field]) : r.id), s, e ?? s, e === null ? 1 : 0], datumId: r.id }];
        });
      out.push({
        type: "custom",
        name: `${lane.name || (y?.title ?? "Events")}${lane.approximated ? " (approx.)" : ""}`,
        xAxisIndex: gi,
        yAxisIndex: gi,
        itemStyle: { color },
        encode: { x: [1, 2], y: 0 },
        data,
        renderItem: (_p: any, api: any) => {
          const row = api.value(0);
          const start = api.coord([api.value(1), row]);
          const end = api.coord([api.value(2), row]);
          const hh = api.size([0, 1])[1] * 0.5;
          const dash = lane.approximated ? [5, 3] : null;
          if (api.value(3) === 1) {
            const r = Math.min(8, hh / 2);
            return {
              type: "path",
              shape: { pathData: `M${start[0]},${start[1] - r}L${start[0] + r},${start[1]}L${start[0]},${start[1] + r}L${start[0] - r},${start[1]}Z` },
              style: { fill: color, stroke: theme.bg, lineWidth: 1 },
            };
          }
          return {
            type: "rect",
            shape: { x: start[0], y: start[1] - hh / 2, width: Math.max(3, end[0] - start[0]), height: hh, r: 3 },
            style: lane.approximated ? { fill: color, opacity: 0.55, stroke: color, lineDash: dash } : { fill: color, opacity: 0.9 },
          };
        },
        ...(li === 0 ? {
          markLine: {
            symbol: "none", silent: true,
            lineStyle: { color: theme.reference, type: "dashed" },
            label: { formatter: "{b}", color: theme.muted, position: "start" },
            data: g.kind === "relative" ? (g.origin ? [{ xAxis: 0, name: g.origin }] : []) : [{ xAxis: Date.now(), name: "Today" }],
          },
        } : {}),
      });
    });
  });
  return {
    ...baseOption(theme),
    legend: { ...baseOption(theme).legend, show: allLanes.length > 1 || !!allLanes[0]?.name },
    tooltip: itemTooltip(viz, lookup, theme, label ? (d) => str(d[label.field]) : y ? (d) => str(d[y.field]) : undefined),
    grid: grids,
    xAxis: xAxes,
    yAxis: yAxes,
    series: out,
  };
}

function humanizeUnit(u: string): string {
  return u.charAt(0).toUpperCase() + u.slice(1);
}

/* ------------------------------------------------------- survival_curve */

export function survivalOption(viz: Visualization, theme: ChartTheme): EOption {
  const rows = rowsOf(viz);
  const { x, y, series, error_low: lo, error_high: hi } = viz.encoding;
  if (!x || !y) return {};
  const lookup = byId(rows);
  const keys = series ? categories(rows, series.field, series.sort) : [""];
  const out: EOption[] = [];
  keys.forEach((key, si) => {
    const color = colorFor(theme, si);
    const name = series ? prettyEnum(key) : y.title;
    const pts = rows
      .filter((r) => !series || str(r[series.field]) === key)
      .map((r) => ({ r, t: num(r[x.field]), s: num(r[y.field]) }))
      .filter((p): p is { r: Datum; t: number; s: number } => p.t !== null && p.s !== null)
      .sort((a, b) => a.t - b.t);
    const segs = pts.slice(1).map((p, i) => {
      const prev = pts[i];
      return {
        value: [prev.t, prev.s, p.t, p.s, p.r.source === "interpolated" ? 1 : 0,
          lo ? num(prev.r[lo.field]) ?? NaN : NaN, hi ? num(prev.r[hi.field]) ?? NaN : NaN],
      };
    });
    if (lo && hi) {
      out.push({
        type: "custom", name, silent: true, z: 1, tooltip: { show: false }, data: segs,
        renderItem: (_p: any, api: any) => {
          const l = api.value(5);
          const h = api.value(6);
          if (!Number.isFinite(l) || !Number.isFinite(h)) return null;
          const a = api.coord([api.value(0), h]);
          const b = api.coord([api.value(2), l]);
          return { type: "rect", shape: { x: a[0], y: a[1], width: b[0] - a[0], height: b[1] - a[1] }, style: { fill: color, opacity: 0.14 } };
        },
      });
    }
    out.push({
      type: "custom", name, z: 2, silent: true, tooltip: { show: false }, data: segs,
      renderItem: (_p: any, api: any) => {
        const a = api.coord([api.value(0), api.value(1)]);
        const b = api.coord([api.value(2), api.value(1)]);
        const c = api.coord([api.value(2), api.value(3)]);
        const dashed = api.value(4) === 1;
        return {
          type: "polyline",
          shape: { points: [a, b, c] },
          style: { stroke: color, lineWidth: 2.5, fill: "none", lineDash: dashed ? [6, 4] : null },
        };
      },
    });
    out.push({
      type: "scatter", name, z: 3, symbolSize: 8,
      itemStyle: { color, borderColor: theme.bg, borderWidth: 1 },
      data: pts.map((p) => ({
        value: [p.t, p.s], datumId: p.r.id,
        ...(p.r.source === "interpolated" ? { symbol: "emptyCircle", symbolSize: 6 } : {}),
      })),
    });
  });
  return {
    ...baseOption(theme),
    legend: { ...baseOption(theme).legend, show: !!series },
    tooltip: itemTooltip(viz, lookup, theme, (d) => (d.source === "interpolated" ? "Interpolated point" : d.source === "reported" ? "Reported point" : "")),
    grid: GRID,
    xAxis: valueAxis(theme, x, viz.meta, { min: 0 }, false),
    yAxis: valueAxis(theme, y, viz.meta, y.scale?.domain ? {} : { min: 0, max: 1 }, false),
    series: out,
  };
}

/* ---------------------------------------- threshold_line / power_curve */

function markLinesAndAreas(viz: Visualization, theme: ChartTheme, xNumeric: boolean): EOption {
  const thresholds = (viz.meta?.thresholds ?? []).filter((t) => (t.axis ?? "y") === "y" || xNumeric);
  const bands = (viz.meta?.bands ?? []).filter((b) => (b.axis ?? "y") === "y" || xNumeric);
  const extra: EOption = {};
  if (thresholds.length) {
    extra.markLine = {
      symbol: "none",
      silent: true,
      lineStyle: { color: theme.threshold, type: "dashed", width: 1.5 },
      label: { color: theme.threshold, formatter: "{b}", position: "insideEndTop" },
      data: thresholds.map((t) => ((t.axis ?? "y") === "x" ? { xAxis: t.value, name: t.label } : { yAxis: t.value, name: t.label })),
    };
  }
  if (bands.length) {
    extra.markArea = {
      silent: true,
      itemStyle: { color: theme.band },
      label: { color: theme.muted, position: "insideTopLeft" },
      data: bands.map((b) => ((b.axis ?? "y") === "x"
        ? [{ xAxis: b.low, name: b.label }, { xAxis: b.high }]
        : [{ yAxis: b.low, name: b.label }, { yAxis: b.high }])),
    };
  }
  return extra;
}

export function thresholdOption(viz: Visualization, theme: ChartTheme): EOption {
  const rows = rowsOf(viz);
  const { x, y, series } = viz.encoding;
  if (!x || !y) return {};
  const lookup = byId(rows);
  const xNumeric = x.type === "quantitative";
  const keys = series ? categories(rows, series.field, series.sort) : [""];
  const cats = xNumeric ? [] : categories(rows, x.field, x.sort ?? viz.meta?.sort, y.field);
  const marks = markLinesAndAreas(viz, theme, xNumeric);
  const yMaxThreshold = Math.max(0, ...(viz.meta?.thresholds ?? []).filter((t) => (t.axis ?? "y") === "y").map((t) => t.value));
  const out = keys.map((key, si) => {
    const subset = rows.filter((r) => !series || str(r[series.field]) === key);
    const data = xNumeric
      ? subset
        .map((r) => ({ xv: num(r[x.field]), yv: num(r[y.field]), r }))
        .filter((p) => p.xv !== null && p.yv !== null)
        .sort((a, b) => a.xv! - b.xv!)
        .map((p) => ({ value: [p.xv, p.yv], datumId: p.r.id }))
      : cats.map((c) => {
        const r = subset.find((row) => str(row[x.field]) === c);
        const v = r ? num(r[y.field]) : null;
        return r && v !== null ? { value: v, datumId: r.id } : { value: null };
      });
    return {
      type: xNumeric ? "line" : "scatter",
      name: series ? prettyEnum(key) : y.title,
      data,
      smooth: viz.type === "power_curve",
      symbolSize: xNumeric ? 6 : 12,
      itemStyle: { color: colorFor(theme, si) },
      lineStyle: { width: 2.5 },
      emphasis: { focus: "series" },
      ...(si === 0 ? marks : {}),
    };
  });
  const yAxis = valueAxis(theme, y, viz.meta, {}, allPositive(rows, [y.field]));
  if (yAxis.max === undefined && yMaxThreshold > 0) {
    const ys = rows.map((r) => num(r[y.field])).filter((v): v is number => v !== null);
    if (yMaxThreshold > Math.max(0, ...ys)) yAxis.max = Math.ceil(yMaxThreshold * 1.1);
  }
  return {
    ...baseOption(theme),
    legend: { ...baseOption(theme).legend, show: !!series },
    tooltip: itemTooltip(viz, lookup, theme),
    grid: { ...GRID, right: 48 },
    xAxis: xNumeric ? valueAxis(theme, x, viz.meta, { scale: true }, allPositive(rows, [x.field]))
      : { ...categoryAxis(theme, x, viz.meta, cats), nameGap: cats.length > 6 ? 64 : 36, axisLabel: { color: theme.muted, rotate: cats.length > 6 ? 30 : 0, formatter: (v: string) => truncate(prettyEnum(v), 22) } },
    yAxis,
    series: out,
  };
}

/* ---------------------------------------------------------- forest_plot */

export function forestOption(viz: Visualization, theme: ChartTheme): EOption {
  const rows = rowsOf(viz);
  const { x, y, error_low: lo, error_high: hi, size, series } = viz.encoding;
  if (!x || !y) return {};
  const lookup = byId(rows);
  const labels = categories(rows, y.field, y.sort);
  const refLine = viz.meta?.reference_line;
  const ref = refLine ? refLine.value : null;
  const log = x.scale?.type === "log" && logFallbacks(viz).length === 0;
  const scale = sizeScaler(rows, size?.field, 8, 20);
  const keys = series ? categories(rows, series.field, series.sort) : [""];
  const vals = rows.flatMap((r) => [num(r[x.field]), lo ? num(r[lo.field]) : null, hi ? num(r[hi.field]) : null])
    .filter((v): v is number => v !== null);
  if (ref !== null) vals.push(ref);
  const minV = vals.length ? Math.min(...vals) : 0;
  const maxV = vals.length ? Math.max(...vals) : 1;
  const out: EOption[] = [];
  keys.forEach((key, si) => {
    const subset = rows.filter((r) => !series || str(r[series.field]) === key);
    const color = colorFor(theme, si);
    if (lo && hi) {
      out.push({
        type: "custom", name: series ? prettyEnum(key) : x.title, silent: true, tooltip: { show: false },
        data: subset
          .map((r) => ({ l: num(r[lo.field]), h: num(r[hi.field]), row: labels.indexOf(str(r[y.field])) }))
          .filter((w) => w.l !== null && w.h !== null)
          .map((w) => ({ value: [w.l, w.h, w.row] })),
        renderItem: (_p: any, api: any) => {
          const a = api.coord([api.value(0), api.value(2)]);
          const b = api.coord([api.value(1), api.value(2)]);
          const cap = 5;
          const style = { stroke: color, lineWidth: 2 };
          return {
            type: "group",
            children: [
              { type: "line", shape: { x1: a[0], y1: a[1], x2: b[0], y2: b[1] }, style },
              { type: "line", shape: { x1: a[0], y1: a[1] - cap, x2: a[0], y2: a[1] + cap }, style },
              { type: "line", shape: { x1: b[0], y1: b[1] - cap, x2: b[0], y2: b[1] + cap }, style },
            ],
          };
        },
      });
    }
    out.push({
      type: "scatter",
      name: series ? prettyEnum(key) : x.title,
      symbol: "rect",
      itemStyle: { color, borderColor: theme.bg, borderWidth: 1 },
      data: subset
        .map((r) => ({ v: num(r[x.field]), r }))
        .filter((p) => p.v !== null)
        .map(({ v, r }) => ({ value: [v, labels.indexOf(str(r[y.field]))], symbolSize: scale(r), datumId: r.id })),
      ...(si === 0 && ref !== null ? {
        markLine: {
          symbol: "none", silent: true,
          lineStyle: { color: theme.reference, type: "solid", width: 1 },
          label: { formatter: refLine?.label ?? formatValue(ref), color: theme.muted, position: "start" },
          data: [{ xAxis: ref }],
        },
      } : {}),
    });
  });
  const pad = (maxV - minV) * 0.08 || 1;
  const logMin = 2 ** Math.floor(Math.log2(Math.max(1e-6, minV)));
  const logMax = 2 ** Math.ceil(Math.log2(Math.max(1e-6, maxV)));
  return {
    ...baseOption(theme),
    legend: { ...baseOption(theme).legend, show: !!series },
    tooltip: itemTooltip(viz, lookup, theme, (d) => str(d[y.field])),
    grid: { ...GRID, left: 16, right: 40 },
    xAxis: valueAxis(theme, x, viz.meta, log
      ? { logBase: 2, min: logMin, max: logMax, axisLabel: { color: theme.muted, formatter: (v: number) => formatValue(v, x.format ?? ".2~f") } }
      : { min: minV - pad, max: maxV + pad }, log),
    yAxis: { ...categoryAxis(theme, y, viz.meta, labels), type: "category", inverse: true, name: "", axisLabel: { color: theme.muted, width: 180, overflow: "truncate" } },
    series: out,
  };
}

/* -------------------------------------------------------------- geo_map */

/** Country-level region/value fields: meta.geo.region_field/value_field, else encoding.x / encoding.y. */
export function geoCountryFields(viz: Visualization): { region: string; value: string } | null {
  const geo = viz.meta?.geo;
  const region = geo?.region_field ?? viz.encoding.x?.field;
  const value = geo?.value_field ?? viz.encoding.y?.field;
  return region && value ? { region, value } : null;
}

/** Country level: sorted horizontal bar of meta.geo.region_field by meta.geo.value_field (no map asset). */
export function geoCountryOption(viz: Visualization, theme: ChartTheme): EOption {
  const geo = viz.meta?.geo;
  const fields = geoCountryFields(viz);
  if (!geo || geo.level !== "country" || !fields) return {};
  const chans = [viz.encoding.x, viz.encoding.y, viz.encoding.label, viz.encoding.series].filter((c): c is Channel => !!c);
  const region = chans.find((c) => c.field === fields.region) ?? { field: fields.region, type: "nominal", title: "Region" };
  const value = chans.find((c) => c.field === fields.value) ?? { field: fields.value, type: "quantitative", title: prettyEnum(fields.value) };
  const sorted: Visualization = {
    ...viz,
    encoding: { ...viz.encoding, x: { ...region, sort: { by: value.field, order: "desc" } }, y: value, series: undefined },
    meta: { ...viz.meta, layout: { orientation: "horizontal", mode: "grouped" } },
  };
  return barOption(sorted, theme);
}

/** Axis range around the points (10° padding, snapped to 10°, clamped to ±limit). */
function extent(vals: (number | null)[], limit: number): [number, number] {
  const v = vals.filter((x): x is number => x !== null);
  if (!v.length) return [-limit, limit];
  const lo = Math.max(-limit, Math.floor((Math.min(...v) - 10) / 10) * 10);
  const hi = Math.min(limit, Math.ceil((Math.max(...v) + 10) / 10) * 10);
  return [lo, hi];
}

/** Site level: scatter of meta.geo.lon_field / lat_field on plain lon/lat axes, sized by encoding.size. */
export function geoSiteOption(viz: Visualization, theme: ChartTheme): EOption {
  const geo = viz.meta?.geo;
  if (!geo || geo.level !== "site" || !geo.lat_field || !geo.lon_field) return {};
  const rows = rowsOf(viz);
  const { series, label, size } = viz.encoding;
  const lookup = byId(rows);
  const keys = series ? categories(rows, series.field, series.sort) : [""];
  // encoding.size (such as trials at the site) scales the points; without it every site is the same size.
  const sizeOf = size ? sizeScaler(rows, size.field, 7, 34) : () => 11;
  const deg = (v: number) => `${v}°`;
  const axis = (name: string, min: number, max: number) => ({
    type: "value", name, nameLocation: "middle", nameGap: 32, min, max, ...axisStyle(theme),
    axisLine: { onZero: false, lineStyle: { color: theme.muted } },
    axisLabel: { color: theme.muted, formatter: deg },
  });
  return {
    ...baseOption(theme),
    legend: { ...baseOption(theme).legend, show: !!series },
    tooltip: itemTooltip(viz, lookup, theme, label ? (d) => str(d[label.field]) : undefined),
    grid: GRID,
    xAxis: axis("Longitude", ...extent(rows.map((r) => num(r[geo.lon_field!])), 180)),
    yAxis: axis("Latitude", ...extent(rows.map((r) => num(r[geo.lat_field!])), 90)),
    series: keys.map((key, si) => ({
      type: "scatter",
      name: series ? prettyEnum(key) : "Sites",
      itemStyle: { color: colorFor(theme, si), borderColor: theme.bg, borderWidth: 1, opacity: size ? 0.75 : 1 },
      data: rows
        .filter((r) => !series || str(r[series.field]) === key)
        .map((r) => ({ lon: num(r[geo.lon_field!]), lat: num(r[geo.lat_field!]), r }))
        .filter((p) => p.lon !== null && p.lat !== null)
        .map((p) => ({ value: [p.lon, p.lat], symbolSize: sizeOf(p.r), datumId: p.r.id })),
    })),
  };
}
