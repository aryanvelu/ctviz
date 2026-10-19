// Plain React renderers: kpi_cards, table, eligibility_checklist, and the data table that sits
// under every chart (the keyboard / screen-reader equivalent of the chart).
import { useMemo, useState } from "react";
import type { Channel, Datum, TableColumn, Visualization } from "../types";
import { isGraphData } from "../types";
import { channelSpec, channelsOf, displayValue, formatChannel, formatValue, humanize, isScalarish, prettyEnum, safeHttpUrl, str, tooltipLines } from "../format";
import { graphFields } from "./options";

interface Props { viz: Visualization; onSelect: (id: string) => void; selectedId?: string }

const rowsOf = (viz: Visualization): Datum[] => (Array.isArray(viz.data) ? viz.data : []);

function citeCount(d: Datum): number {
  return d.citations_total ?? d.citations?.length ?? 0;
}

function CiteButton({ d, onSelect, label }: { d: Datum; onSelect: (id: string) => void; label?: string }) {
  const n = citeCount(d);
  return (
    <button type="button" className="cite-btn" onClick={(e) => { e.stopPropagation(); onSelect(d.id); }}
      aria-label={`${n ? `Show ${n} source${n === 1 ? "" : "s"}` : "Show source note"}${label ? ` for ${label}` : ""}`}>
      {n ? "Sources" : "Note"}
    </button>
  );
}

/* ------------------------------------------------------------------ kpi_cards */

/** Value text for a card. Reserved `kind` (number | text | date | enum | list) picks the formatting. */
function kpiValue(d: Datum, ch: Channel | undefined, viz: Visualization): string {
  if (!ch) return "";
  const v = d[ch.field];
  switch (d.kind) {
    case "list": return Array.isArray(v) ? v.map((x) => displayValue(x)).join(", ") : displayValue(v);
    case "enum": return prettyEnum(formatValue(v));
    case "date": return formatValue(v, ch.time_format);
    case "text": return displayValue(v);
    default: return typeof v === "number" ? formatChannel(d, ch, viz.meta) : displayValue(v);
  }
}

/** Values this short keep the big number font; longer text is set at body size. */
const KPI_SHORT = 12;
/** Tooltip fields shown as a quiet line without their name ("Unit: kg" reads as just "kg"). */
const QUIET_FIELDS = new Set(["unit", "units", "detail", "details", "note", "description"]);

/** The quiet line under a card's value, built from encoding.tooltip fields. */
function kpiSubtitle(d: Datum, viz: Visualization): string {
  const enc = viz.encoding;
  const quiet = (enc.tooltip ?? []).filter((f) => QUIET_FIELDS.has(f.toLowerCase()) && d[f] !== undefined && d[f] !== null && d[f] !== "");
  const named = (enc.tooltip ?? []).filter((f) => !QUIET_FIELDS.has(f.toLowerCase()));
  return [...quiet.map((f) => displayValue(d[f])), ...tooltipLines(d, named, viz.meta, enc)].join(" · ");
}

export function KpiCards({ viz, onSelect, selectedId }: Props) {
  const rows = rowsOf(viz);
  const enc = viz.encoding;
  const labelCh = enc.label;
  const valueCh = enc.y;
  const cols = viz.meta?.layout?.columns;
  return (
    <div className="kpi-grid" style={cols ? { gridTemplateColumns: `repeat(auto-fill, minmax(min(100%, max(220px, ${Math.floor(100 / cols) - 2}%)), 1fr))` } : undefined}>
      {rows.map((d) => {
        const value = kpiValue(d, valueCh, viz);
        const short = value.length <= KPI_SHORT;
        const sub = kpiSubtitle(d, viz);
        const label = labelCh ? str(d[labelCh.field]) : d.id;
        return (
          <button type="button" key={d.id} className={`kpi-card ${selectedId === d.id ? "selected" : ""}`} onClick={() => onSelect(d.id)}
            aria-label={`${label}: ${value}. Show sources`}>
            <span className="kpi-label">{label}</span>
            <span className={short ? "kpi-value" : "kpi-value kpi-text"} title={short ? undefined : value}>{value}</span>
            {sub && <span className="kpi-sub">{sub}</span>}
            {d.cite_note && !d.citations?.length && !d.nct_ids?.length
              ? <span className="kpi-sub">{d.cite_note}</span>
              : <span className="kpi-cites">Sources</span>}
          </button>
        );
      })}
    </div>
  );
}

/* ---------------------------------------------------------------------- table */

/** Columns: meta.columns, else the encoded channels plus encoding.tooltip fields. */
export function tableColumns(viz: Visualization): TableColumn[] {
  if (viz.meta?.columns?.length) return viz.meta.columns;
  const enc = viz.encoding;
  const seen = new Set<string>();
  const cols: TableColumn[] = [];
  for (const c of channelsOf(enc)) {
    if (seen.has(c.field)) continue;
    seen.add(c.field);
    cols.push({ field: c.field, title: c.title, type: c.type, format: channelSpec(c) });
  }
  for (const f of enc.tooltip ?? []) {
    if (!seen.has(f)) { seen.add(f); cols.push({ field: f, title: humanize(f), type: "nominal" }); }
  }
  const geo = viz.meta?.geo;
  for (const f of [geo?.region_field, geo?.value_field, geo?.lat_field, geo?.lon_field]) {
    if (f && !seen.has(f)) { seen.add(f); cols.push({ field: f, title: humanize(f), type: f === geo?.region_field ? "nominal" : "quantitative" }); }
  }
  for (const l of viz.meta?.lanes ?? []) {
    for (const f of [l.start_field, l.end_field]) {
      if (f && !seen.has(f)) { seen.add(f); cols.push({ field: f, title: humanize(f), type: l.axis.kind === "calendar" ? "temporal" : "quantitative" }); }
    }
  }
  return cols;
}

interface TableProps extends Props { rowsOverride?: Datum[]; columnsOverride?: TableColumn[]; caption?: string }

export function DataTable({ viz, onSelect, selectedId, rowsOverride, columnsOverride, caption }: TableProps) {
  const rows = rowsOverride ?? rowsOf(viz);
  const cols = useMemo(() => columnsOverride ?? tableColumns(viz), [viz, columnsOverride]);
  const [sort, setSort] = useState<{ field: string; dir: 1 | -1 } | null>(null);
  const sorted = useMemo(() => {
    if (!sort) return rows;
    return [...rows].sort((a, b) => {
      const va = a[sort.field];
      const vb = b[sort.field];
      if (va === undefined && vb !== undefined) return 1;
      if (vb === undefined && va !== undefined) return -1;
      if (typeof va === "number" && typeof vb === "number") return sort.dir * (va - vb);
      return sort.dir * str(va).localeCompare(str(vb), undefined, { numeric: true });
    });
  }, [rows, sort]);
  const units = viz.meta?.units ?? {};
  const firstField = cols[0]?.field;
  return (
    <div className="table-wrap">
      <table className="data-table">
        {caption && <caption className="sr-only">{caption}</caption>}
        <thead>
          <tr>
            {cols.map((c) => (
              <th key={c.field} scope="col" aria-sort={sort?.field === c.field ? (sort.dir === 1 ? "ascending" : "descending") : "none"}>
                <button type="button" className="th-btn" onClick={() => setSort((s) => (s?.field === c.field ? { field: c.field, dir: (s.dir * -1) as 1 | -1 } : { field: c.field, dir: 1 }))}>
                  {c.title}{units[c.field] ? ` (${units[c.field]})` : ""}
                  {sort?.field === c.field ? (sort.dir === 1 ? " ▲" : " ▼") : ""}
                </button>
              </th>
            ))}
            <th scope="col">Sources</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((d) => (
            <tr key={d.id} className={selectedId === d.id ? "selected" : ""} onClick={() => onSelect(d.id)}>
              {cols.map((c) => (
                <td key={c.field} className={c.type === "quantitative" ? "num" : ""}>
                  {renderCell(d[c.field], c)}
                </td>
              ))}
              <td><CiteButton d={d} onSelect={onSelect} label={firstField ? str(d[firstField]) : d.id} /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function renderCell(v: unknown, c: TableColumn) {
  if (v === undefined || v === null) return <span className="muted">—</span>;
  const s = formatValue(v, c.format);
  if (typeof v === "string" && /^NCT\d{8}$/.test(v)) {
    const href = safeHttpUrl(`https://clinicaltrials.gov/study/${v}`);
    return <a href={href} target="_blank" rel="noreferrer" onClick={(e) => e.stopPropagation()}>{v}</a>;
  }
  return c.type === "quantitative" ? s : displayValue(v, c.format);
}

/** The data behind any chart, as keyboard-accessible tables (nodes and edges for graph types). */
export function ChartDataTable({ viz, onSelect, selectedId }: Props) {
  if (isGraphData(viz.data)) {
    const { ne, ee } = graphFields(viz);
    const nodeCols: TableColumn[] = [
      { field: ne.label, title: "Node", type: "nominal" },
      ...(ne.type ? [{ field: ne.type, title: "Type", type: "nominal" as const }] : []),
      ...(ne.size ? [{ field: ne.size, title: humanize(ne.size), type: "quantitative" as const }] : []),
    ];
    const edgeCols: TableColumn[] = [
      { field: ee.source, title: "From", type: "nominal" },
      { field: ee.target, title: "To", type: "nominal" },
      ...(ee.weight ? [{ field: ee.weight, title: humanize(ee.weight), type: "quantitative" as const }] : []),
    ];
    const label = new Map(viz.data.nodes.map((n) => [n.id, str(n[ne.label])]));
    const edges = viz.data.edges.map((e) => ({ ...e, [ee.source]: label.get(str(e[ee.source])) ?? e[ee.source], [ee.target]: label.get(str(e[ee.target])) ?? e[ee.target] }));
    return (
      <>
        <DataTable viz={viz} onSelect={onSelect} selectedId={selectedId} rowsOverride={viz.data.nodes} columnsOverride={nodeCols} caption="Nodes" />
        <DataTable viz={viz} onSelect={onSelect} selectedId={selectedId} rowsOverride={edges} columnsOverride={edgeCols} caption="Edges" />
      </>
    );
  }
  return <DataTable viz={viz} onSelect={onSelect} selectedId={selectedId} caption={viz.title} />;
}

/* ------------------------------------------------------ eligibility_checklist */

const STATUS_LABEL: Record<string, string> = {
  met: "Met", unmet: "Not met", needs_review: "Needs review", unknown: "Unknown",
};
const SECTION_ORDER = ["structured", "inclusion", "exclusion", "unknown"];
const SECTION_TITLE: Record<string, string> = {
  structured: "Registry fields", inclusion: "Inclusion criteria", exclusion: "Exclusion criteria", unknown: "Other criteria",
};
const EVALUATED_BY: Record<string, string> = { rule: "checked by rule", llm: "checked by AI — verify", none: "not evaluated" };

/** Checklist: text from encoding.label; reserved keys section / status / evaluated_by / parent_text. */
export function Checklist({ viz, onSelect, selectedId }: Props) {
  const rows = rowsOf(viz);
  const enc = viz.encoding;
  const textField = enc.label?.field;
  const present = [...new Set(rows.map((r) => str(r.section) || "unknown"))];
  const groups = [...SECTION_ORDER.filter((s) => present.includes(s)), ...present.filter((s) => !SECTION_ORDER.includes(s))];
  const summary = scalarEntries(viz.meta?.summary);
  return (
    <div className="checklist">
      {summary.length > 0 && (
        <div className="checklist-summary" aria-label="Summary">
          {summary.map(([k, v]) => (
            <span key={k} className={`badge status-${k}`}>{STATUS_LABEL[k] ?? humanize(k)}: {formatValue(v)}</span>
          ))}
        </div>
      )}
      {groups.map((g) => {
        const items = rows.filter((r) => (str(r.section) || "unknown") === g);
        let lastParent = "";
        return (
          <section key={g} className="checklist-group">
            <h4>{SECTION_TITLE[g] ?? humanize(g)}</h4>
            <ul>
              {items.map((d) => {
                const status = str(d.status) || "unknown";
                const by = str(d.evaluated_by);
                const parent = str(d.parent_text);
                const showParent = parent && parent !== lastParent;
                lastParent = parent;
                const extra = tooltipLines(d, enc.tooltip, viz.meta, enc);
                return (
                  <li key={d.id} className={selectedId === d.id ? "selected" : ""}>
                    {showParent && <p className="crit-parent">{parent}</p>}
                    <button type="button" className={`crit ${parent ? "nested" : ""}`} onClick={() => onSelect(d.id)}>
                      <span className={`badge status-${status}`}>{STATUS_LABEL[status] ?? humanize(status)}</span>
                      <span className="crit-text">{textField ? str(d[textField]) : d.id}</span>
                      {extra.map((l) => <span key={l} className="crit-reason">{l}</span>)}
                      <span className="crit-meta">
                        {by && <span className={`tag eval-${by}`}>{EVALUATED_BY[by] ?? by}</span>}
                        <span className="tag link-tag">Sources</span>
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
          </section>
        );
      })}
    </div>
  );
}

/** Key/value pairs safe to print: scalars and flat lists of scalars only. */
export function scalarEntries(o: Record<string, unknown> | undefined): [string, unknown][] {
  return o ? Object.entries(o).filter(([, v]) => isScalarish(v)) : [];
}
