// Value formatting and datum access helpers shared by all renderers.
import { format as d3format } from "d3-format";
import type { Channel, Datum, Encoding, Sort, VizMeta } from "./types";

const fmtCache = new Map<string, ((n: number) => string) | null>();

function numberFormatter(spec: string): ((n: number) => string) | null {
  if (!fmtCache.has(spec)) {
    try {
      fmtCache.set(spec, d3format(spec));
    } catch {
      fmtCache.set(spec, null);
    }
  }
  return fmtCache.get(spec) ?? null;
}

/** d3-time-format style strings contain %<letter>; d3-format percent strings end in a bare %. */
export function isTimeFormat(spec: string | undefined): boolean {
  return !!spec && /%[a-zA-Z]/.test(spec);
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** Minimal d3-time-format subset (%Y %y %m %d %b %B %q) over ISO-ish date strings ("2019", "2019-06", "2019-06-04"). */
export function formatDate(value: unknown, spec: string): string {
  const s = String(value);
  const m = /^(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?/.exec(s);
  if (!m) return s;
  const [, y, mo, d] = m;
  return spec.replace(/%([a-zA-Z])/g, (_all, t: string) => {
    switch (t) {
      case "Y": return y;
      case "y": return y.slice(2);
      case "m": return mo ?? "01";
      case "d": return d ?? "01";
      case "b": return MONTHS[Number(mo ?? 1) - 1] ?? "";
      case "B": return MONTHS[Number(mo ?? 1) - 1] ?? "";
      case "q": return String(Math.ceil(Number(mo ?? 1) / 3));
      default: return "";
    }
  });
}

/** Format a raw value using a d3-format / d3-time-format spec when given. */
export function formatValue(value: unknown, spec?: string): string {
  if (value === null || value === undefined) return "—";
  if (spec) {
    if (isTimeFormat(spec)) return formatDate(value, spec);
    if (typeof value === "number") {
      const f = numberFormatter(spec);
      if (f) return f(value);
    }
  }
  if (typeof value === "number") {
    return Number.isInteger(value) ? value.toLocaleString("en-US") : String(Math.round(value * 1000) / 1000);
  }
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (Array.isArray(value)) return value.map((v) => formatValue(v)).join(", ");
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

/** Unit for a channel: channel.unit, else meta.units[field]. Percent formats imply no extra unit. */
export function unitFor(ch: Channel | undefined, meta?: VizMeta): string {
  if (!ch) return "";
  const u = ch.unit ?? meta?.units?.[ch.field] ?? "";
  return u;
}

/** The spec to format a channel's values with: time_format for temporal channels, else format. */
export function channelSpec(ch: Channel | undefined): string | undefined {
  if (!ch) return undefined;
  return ch.type === "temporal" ? ch.time_format ?? ch.format : ch.format;
}

/** Format a channel's value for display, including its unit. */
export function formatChannel(d: Datum, ch: Channel | undefined, meta?: VizMeta): string {
  if (!ch) return "";
  const v = d[ch.field];
  const s = formatValue(v, channelSpec(ch));
  const u = unitFor(ch, meta);
  if (!u || v === null || v === undefined) return s;
  return u === "%" ? (s.endsWith("%") ? s : `${s}%`) : `${s} ${u}`;
}

export function axisName(ch: Channel | undefined, meta?: VizMeta): string {
  if (!ch) return "";
  const u = unitFor(ch, meta);
  return u && !ch.title.toLowerCase().includes(u.toLowerCase()) ? `${ch.title} (${u})` : ch.title;
}

/** Axis label formatter from a channel spec. */
export function axisFormatter(ch: Channel | undefined): ((v: number | string) => string) | undefined {
  const spec = channelSpec(ch);
  if (!spec) return undefined;
  return (v) => {
    if (typeof v === "string" && !isTimeFormat(spec)) {
      const n = Number(v);
      return Number.isFinite(n) && v.trim() !== "" ? formatValue(n, spec) : v;
    }
    return formatValue(v, spec);
  };
}

export function num(v: unknown): number | null {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (typeof v === "string" && v.trim() !== "" && Number.isFinite(Number(v))) return Number(v);
  return null;
}

export function str(v: unknown): string {
  return v === null || v === undefined ? "" : String(v);
}

/**
 * Unique category values of a field, ordered by sort.values, else sort.by/order (ascending unless
 * order is "desc"), else first appearance.
 */
export function categories(rows: Datum[], field: string, sort?: Sort, valueField?: string): string[] {
  const seen: string[] = [];
  const set = new Set<string>();
  for (const r of rows) {
    const k = str(r[field]);
    if (!set.has(k)) {
      set.add(k);
      seen.push(k);
    }
  }
  if (sort?.values?.length) {
    const order = sort.values.filter((v) => set.has(v));
    const rest = seen.filter((v) => !sort.values!.includes(v));
    return [...order, ...rest];
  }
  if (sort?.by || sort?.order) {
    const by = sort.by ?? valueField;
    const dir = sort.order === "desc" ? -1 : 1;
    if (by && by !== field) {
      const totals = new Map<string, number>();
      for (const r of rows) totals.set(str(r[field]), (totals.get(str(r[field])) ?? 0) + (num(r[by]) ?? 0));
      return [...seen].sort((a, b) => dir * ((totals.get(a) ?? 0) - (totals.get(b) ?? 0)));
    }
    return [...seen].sort((a, b) => dir * a.localeCompare(b, undefined, { numeric: true }));
  }
  return seen;
}

/** All channels of an encoding, in display order. */
export function channelsOf(enc: Encoding): Channel[] {
  return [enc.label, enc.x, enc.x2, enc.y, enc.y2, enc.series, enc.size, enc.error_low, enc.error_high]
    .filter((c): c is Channel => !!c);
}

/** Extra tooltip lines from encoding.tooltip, formatted with the matching channel's format when there is one. */
export function tooltipLines(d: Datum, fields: string[] | undefined, meta?: VizMeta, enc?: Encoding): string[] {
  const chans = enc ? channelsOf(enc) : [];
  return (fields ?? [])
    .filter((f) => d[f] !== undefined)
    .map((f) => {
      const ch = chans.find((c) => c.field === f);
      if (ch) return `${ch.title}: ${typeof d[f] === "string" ? prettyEnum(formatChannel(d, ch, meta)) : formatChannel(d, ch, meta)}`;
      const u = meta?.units?.[f];
      return `${humanize(f)}: ${displayValue(d[f])}${u ? ` ${u}` : ""}`;
    });
}

/** Only http(s) URLs may become links; anything else (javascript:, data:, relative junk) is rejected. */
export function safeHttpUrl(u: unknown): string | undefined {
  if (typeof u !== "string" || !u.trim()) return undefined;
  try {
    const url = new URL(u.trim());
    return url.protocol === "https:" || url.protocol === "http:" ? url.href : undefined;
  } catch {
    return undefined;
  }
}

/** True for values safe to show as a plain key/value entry (scalars, or flat lists of scalars). */
export function isScalarish(v: unknown): boolean {
  const scalar = (x: unknown) => x === null || ["string", "number", "boolean"].includes(typeof x);
  return scalar(v) || (Array.isArray(v) && v.every(scalar));
}

export function humanize(s: string): string {
  const t = s.replace(/_/g, " ");
  return t.charAt(0).toUpperCase() + t.slice(1);
}

/** Short registry codes that are words, not acronyms (longer ones are caught by the length rule below). */
const SHORT_ENUMS: Record<string, string> = { DRUG: "Drug", NA: "Not applicable", MALE: "Male", NONE: "None" };

/** Display names for visualization types ("kpi_cards" -> "KPI cards"). */
const VIZ_LABELS: Record<string, string> = { kpi_cards: "KPI cards", geo_map: "Map", eligibility_checklist: "Eligibility checklist" };

export function vizLabel(type: string): string {
  return VIZ_LABELS[type] ?? humanize(type);
}

/** Pretty enum values: "ACTIVE_NOT_RECRUITING" -> "Active not recruiting". Leaves mixed-case text alone. */
export function prettyEnum(v: string): string {
  if (SHORT_ENUMS[v]) return SHORT_ENUMS[v];
  if (/^PHASE\d$/.test(v)) return `Phase ${v.slice(5)}`;
  if (/^EARLY_PHASE\d$/.test(v)) return `Early phase ${v.slice(11)}`;
  if (/^[A-Z0-9_]+$/.test(v) && v.includes("_")) return humanize(v.toLowerCase());
  if (/^[a-z0-9]+(_[a-z0-9]+)+$/.test(v)) return humanize(v);
  // Single upper-case words: long ones are enum values (COMPLETED); short ones are likely acronyms (NASH, DRUG kept).
  if (/^[A-Z]+$/.test(v) && v.length >= 5) return humanize(v.toLowerCase());
  return v;
}

/** formatValue, with registry enum codes (RECRUITING, PHASE2, ESTIMATED) shown as words. */
export function displayValue(value: unknown, spec?: string): string {
  if (typeof value === "string") return prettyEnum(formatValue(value, spec));
  if (Array.isArray(value)) return value.map((v) => displayValue(v)).join(", ");
  return formatValue(value, spec);
}

export function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
}
