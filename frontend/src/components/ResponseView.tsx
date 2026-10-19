// Everything below the query bar for one response (public JSON, schema 2.0):
// status ok -> chart, answer, follow-ups, alternatives; otherwise the clarification card.
// Then the collapsible response JSON, exactly as received. meta.notes stay in the JSON only.
import { Fragment, useCallback, useMemo, useState, type ReactNode } from "react";
import type { Channel, Clarification, Datum, VisualizeResponse, Visualization, VizType } from "../types";
import { formatChannel, formatValue, humanize, prettyEnum, str, vizLabel } from "../format";
import { toRenderViz } from "../adapt";
import { findDatum, Renderer } from "../viz/Renderer";
import { graphFields } from "../viz/options";
import { isGraphData } from "../types";
import { CitationPanel } from "./CitationPanel";

/** Turn NCT ids in text into study links (handles "[NCT01234567]" and "[NCT1, NCT2]"). */
export function linkifyNct(text: string): ReactNode[] {
  const parts = text.split(/(NCT\d{8})/g);
  return parts.map((p, i) => (/^NCT\d{8}$/.test(p)
    ? <a key={i} href={`https://clinicaltrials.gov/study/${p}`} target="_blank" rel="noreferrer">{p}</a>
    : <Fragment key={i}>{p}</Fragment>));
}

/** A short human title for a datum, built only from encoded fields. */
function datumTitle(d: Datum, viz: Visualization): string {
  const enc = viz.encoding;
  if (isGraphData(viz.data)) {
    const { ne, ee } = graphFields(viz);
    const nodes = viz.data.nodes;
    if (viz.data.edges.some((e) => e.id === d.id)) {
      const name = (id: unknown) => str(nodes.find((n) => n.id === str(id))?.[ne.label] ?? id);
      return `${name(d[ee.source])} → ${name(d[ee.target])}`;
    }
    return str(d[ne.label]) || "This value";
  }
  if (typeof d.label === "string" && d.id.startsWith("meta.thresholds[")) return d.label;
  const chans = [enc.label, enc.x, enc.series, enc.y, enc.size].filter((c): c is Channel => !!c);
  const parts: string[] = [];
  for (const c of chans) {
    const v = d[c.field];
    if (v === undefined || v === null || typeof v === "object" || parts.length >= 3) continue;
    const s = typeof v === "string" && c.type !== "temporal" ? prettyEnum(formatValue(v)) : formatChannel(d, c, viz.meta);
    if (!parts.includes(s)) parts.push(s);
  }
  return parts.join(" · ") || "This value";
}

export interface ResponseActions {
  /** Re-run the last request with a preferred visualization type. */
  onAlternative?: (t: VizType) => void;
  /** Ask a new question (follow-up chip). */
  onAsk?: (question: string) => void;
  /** Put a suggestion into the question box for editing. */
  onSuggest?: (text: string) => void;
  /** Re-ask the same question about one candidate trial. */
  onPickTrial?: (nctId: string) => void;
}

interface Props extends ResponseActions {
  response: VisualizeResponse;
  /** The body exactly as received; defaults to the pretty-printed response. */
  raw?: string;
  busy?: boolean;
}

const noop = () => {};

export function ResponseView({ response, raw, busy = false, onAlternative = noop, onAsk = noop, onSuggest = noop, onPickTrial = noop }: Props) {
  const [selected, setSelected] = useState<string | null>(null);
  const viz = useMemo(() => toRenderViz(response), [response]);
  const onSelect = useCallback((id: string) => setSelected(id), []);
  const datum = viz && selected ? findDatum(viz, selected) : undefined;
  const meta = response.meta;
  const status = response.status ?? "ok";
  return (
    <div className={`response ${datum ? "with-panel" : ""}`}>
      <div className="response-main">
        {status !== "ok" && (
          <p className={`status-line status-${status}`} data-testid="status">
            <span className="badge">{humanize(status)}</span>
          </p>
        )}
        {status === "ok" && viz ? (
          <>
            <Renderer viz={viz} onSelect={onSelect} selectedId={selected ?? undefined} />
            {response.answer_summary && (
              <section className="answer" aria-label="Answer">
                <p>{linkifyNct(response.answer_summary)}</p>
              </section>
            )}
            {!!meta.alternatives?.length && (
              <div className="alternatives" role="group" aria-label="Alternative visualizations">
                <span className="alt-label">View as</span>
                <button type="button" className="chip active" aria-pressed="true" disabled>{vizLabel(viz.type)}</button>
                {meta.alternatives.map((a) => (
                  <button key={a.type} type="button" className="chip" title={a.reason} disabled={busy} onClick={() => onAlternative(a.type)}>
                    {vizLabel(a.type)}
                  </button>
                ))}
              </div>
            )}
            <FollowUps items={meta.follow_ups ?? []} disabled={busy} onPick={onAsk} />
          </>
        ) : (
          <ClarificationCard status={status} c={response.clarification} busy={busy} onSuggest={onSuggest} onPickTrial={onPickTrial} />
        )}
        <ResponseJson response={response} raw={raw} />
        {meta.data_timestamp && <p className="muted small footer-meta">Data from ClinicalTrials.gov as of {meta.data_timestamp.slice(0, 10)}</p>}
      </div>
      {datum && viz && (
        <CitationPanel key={datum.id} datum={datum} title={datumTitle(datum, viz)} requestId={meta.request_id} studyUrl={meta.citation_policy?.study_url} onClose={() => setSelected(null)}
          onSelectDatum={onSelect} datumLabel={(id) => { const d = findDatum(viz, id); return d ? datumTitle(d, viz) : undefined; }} />
      )}
    </div>
  );
}

/** "Ask next": each follow-up question is a card; clicking one asks it. */
function FollowUps({ items, onPick, disabled }: { items: string[]; onPick: (s: string) => void; disabled?: boolean }) {
  if (!items.length) return null;
  return (
    <section className="follow-ups" aria-label="Follow-up questions">
      <h3 className="section-title">Ask next</h3>
      <div className="follow-up-list">
        {items.map((q) => (
          <button key={q} type="button" className="follow-up" disabled={disabled} onClick={() => onPick(q)}>
            <span>{q}</span>
            <span className="follow-up-arrow" aria-hidden="true">→</span>
          </button>
        ))}
      </div>
    </section>
  );
}

const REASON_TITLE: Record<string, string> = {
  unrecognized_entity: "Nothing in ClinicalTrials.gov matched part of the question",
  no_entity: "The question needs a drug, condition, sponsor or trial",
  subjective_question: "The registry cannot answer this kind of question",
  trial_required: "This question is about one specific trial",
  ambiguous_entity: "A name in the question is ambiguous",
};

export function ClarificationCard({ status, c, busy, onSuggest, onPickTrial }: {
  status: string; c?: Clarification; busy?: boolean; onSuggest: (s: string) => void; onPickTrial: (id: string) => void;
}) {
  return (
    <section className={`clarification ${status}`} aria-label="Clarification" role="region">
      <p className="eyebrow">{status === "unanswerable" ? "Unanswerable" : "Needs clarification"} · no chart</p>
      <h2>{c ? REASON_TITLE[c.reason] ?? humanize(c.reason) : "No chart was produced"}</h2>
      {c?.message && <p>{linkifyNct(c.message)}</p>}
      {!!c?.unmatched_terms?.length && (
        <p className="small">Unmatched: {c.unmatched_terms.map((t) => <code key={t} className="unmatched">{t}</code>)}</p>
      )}
      {!!c?.suggestions?.length && (
        <div className="chip-row" role="group" aria-label="Suggestions">
          <span className="muted small">Try instead (click to edit in the question box):</span>
          {c.suggestions.map((s) => <button key={s} type="button" className="chip" disabled={busy} onClick={() => onSuggest(s)}>{s}</button>)}
        </div>
      )}
      {!!c?.options?.length && (
        <div className="trial-options">
          <p className="small muted">Pick a trial to ask the same question about it:</p>
          <ul className="plain">
            {c.options.map((o) => (
              <li key={o.nct_id}>
                <button type="button" className="option-row" disabled={busy} onClick={() => onPickTrial(o.nct_id)}>
                  <code>{o.nct_id}</code> <span>{o.title}</span> {o.status && <span className="tag">{humanize(o.status.toLowerCase())}</span>}
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}

/** The body of POST /v1/visualize, collapsed by default. */
function ResponseJson({ response, raw }: { response: VisualizeResponse; raw?: string }) {
  const text = useMemo(() => prettyRaw(raw, response), [raw, response]);
  const [copied, setCopied] = useState(false);
  const copy = () => {
    void navigator.clipboard?.writeText(text).then(() => { setCopied(true); window.setTimeout(() => setCopied(false), 1500); }, noop);
  };
  return (
    <details className="panel json-panel">
      <summary>Response JSON</summary>
      <p className="small muted">Exactly what <code>POST /v1/visualize</code> returned for request <code>{response.meta.request_id}</code> ({formatValue(text.length)} characters). <button type="button" className="link-btn small" onClick={copy}>{copied ? "Copied" : "Copy"}</button></p>
      <pre className="raw-json" data-testid="public-json">{text}</pre>
    </details>
  );
}

/** Pretty-print the body as received (key order and values untouched); fall back to the parsed object. */
function prettyRaw(raw: string | undefined, response: VisualizeResponse): string {
  if (raw) {
    try {
      return JSON.stringify(JSON.parse(raw), null, 2);
    } catch {
      return raw;
    }
  }
  return JSON.stringify(response, null, 2);
}
