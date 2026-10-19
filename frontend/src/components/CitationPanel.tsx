// Side panel listing the sources behind one value: the trials, and an excerpt from each record.
// Mount it with key={datum.id} so state never leaks between datums; an in-flight "show all" fetch is
// aborted and ignored once the datum changes.
import { Fragment, useEffect, useRef, useState, type KeyboardEvent } from "react";
import { fetchAllCitations, type DatumSources } from "../api";
import { prettyEnum, safeHttpUrl } from "../format";
import type { Citation, Datum } from "../types";

interface Props {
  datum: Datum;
  title: string;
  requestId: string;
  onClose: () => void;
  /** meta.citation_policy.study_url, e.g. "https://clinicaltrials.gov/study/{nct_id}". */
  studyUrl?: string;
  /** Opens another datum's sources (used for the anchors of an interpolated value). */
  onSelectDatum?: (id: string) => void;
  /** Display name of another datum by id; undefined when it is not in the response. */
  datumLabel?: (id: string) => string | undefined;
}

const DEFAULT_STUDY_URL = "https://clinicaltrials.gov/study/{nct_id}";
const NCT_RE = /^NCT\d{8}$/;

/** Study page for an NCT id from the policy template (falls back to the default if the template is unsafe). */
export function studyLink(nct: string | undefined, template = DEFAULT_STUDY_URL): string | undefined {
  if (!nct || !NCT_RE.test(nct)) return undefined;
  return safeHttpUrl(template.replace("{nct_id}", nct)) ?? DEFAULT_STUDY_URL.replace("{nct_id}", nct);
}

export function CitationPanel({ datum, title, requestId, onClose, studyUrl, onSelectDatum, datumLabel }: Props) {
  const inline = datum.citations ?? [];
  const total = datum.citations_total ?? inline.length;
  // One fetch of the all-citations endpoint fills both the excerpts and the complete trial list.
  const [all, setAll] = useState<DatumSources | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const panel = useRef<HTMLElement>(null);
  const inflight = useRef<AbortController | null>(null);

  useEffect(() => {
    panel.current?.focus();
    return () => inflight.current?.abort();
  }, []);

  const shown = all?.citations ?? inline;
  const more = all ? 0 : total - inline.length;

  async function loadMore() {
    if (all) return;
    inflight.current?.abort();
    const ctl = new AbortController();
    inflight.current = ctl;
    setLoading(true);
    setError(null);
    try {
      const sources = await fetchAllCitations(requestId, datum.id, ctl.signal);
      if (!ctl.signal.aborted) setAll(sources);
    } catch (e) {
      if (!ctl.signal.aborted) setError((e as Error).message || "Could not load the excerpts.");
    } finally {
      if (!ctl.signal.aborted) setLoading(false);
    }
  }

  const onKeyDown = (e: KeyboardEvent) => {
    if (e.key === "Escape") {
      e.stopPropagation();
      onClose();
    }
  };

  // Excerpts that point at one trial; registry totals (no nct_id) are covered by the search link.
  const excerpts = shown.filter((c) => c.kind !== "derived_count" || !datum.query_url);
  const groups = groupByTrial(excerpts);
  const ids = all?.nct_ids?.length ? all.nct_ids : datum.nct_ids ?? [];
  const idsTotal = all?.nct_ids?.length ? all.nct_ids_total ?? all.nct_ids.length : datum.nct_ids_total;
  // A successful fetch without nct_ids (registry counts, or a response stored before schema 2.2).
  const fullListMissing = !!all && !all.nct_ids?.length;
  const anchors = (datum.derived_from ?? []).filter((id) => typeof id === "string" && id !== datum.id);

  return (
    <aside ref={panel} className="citation-panel" aria-label="Sources" role="complementary" tabIndex={-1} onKeyDown={onKeyDown}>
      <header>
        <div>
          <p className="eyebrow">Sources</p>
          <h3>{title}</h3>
        </div>
        <button type="button" className="icon-btn" onClick={onClose} aria-label="Close sources">×</button>
      </header>
      <QueryLink url={datum.query_url} />
      <TrialList ids={ids} total={idsTotal} studyUrl={studyUrl} onLoadAll={fullListMissing ? undefined : loadMore} loading={loading} unavailable={fullListMissing} />
      {anchors.length > 0 && (
        <p className="derived-from">
          Interpolated between:{" "}
          {anchors.map((id, i) => (
            <Fragment key={id}>
              {i > 0 && " and "}
              {onSelectDatum
                ? <button type="button" className="link-btn" onClick={() => onSelectDatum(id)}>{datumLabel?.(id) ?? id}</button>
                : <span>{datumLabel?.(id) ?? id}</span>}
            </Fragment>
          ))}
        </p>
      )}
      {datum.cite_note && <p className="cite-note">{datum.cite_note}</p>}
      {excerpts.length === 0 && !datum.cite_note && !ids.length && !datum.query_url && !anchors.length && <p className="muted">No sources for this value.</p>}
      {excerpts.length > 0 && <p className="section-label">From the trial records</p>}
      <ul className="citations">
        {groups.map((g, i) => <CitationItem key={`${g.nct}-${i}`} group={g} studyUrl={studyUrl} />)}
      </ul>
      {more > 0 && (
        <div className="more">
          <button type="button" className="link-btn" onClick={loadMore} disabled={loading}>
            {loading ? "Loading…" : `Show all ${total.toLocaleString("en-US")} excerpts`}
          </button>
        </div>
      )}
      {error && <p className="error small" role="alert">{error}</p>}
    </aside>
  );
}

const TRIALS_SHOWN = 12;

/**
 * datum.nct_ids: the trials behind the value, each linking to its study page. The heading counts
 * `total` (datum.nct_ids_total, schema 2.2+) when the server listed only the first few ids; then
 * "Show all" calls `onLoadAll` (the all-citations endpoint), and the parent passes the complete ids.
 */
export function TrialList({ ids, total, studyUrl, onLoadAll, loading = false, unavailable = false }: {
  ids: string[]; total?: number; studyUrl?: string; onLoadAll?: () => void; loading?: boolean;
  /** The endpoint answered but had no complete list: say so instead of offering "Show all". */
  unavailable?: boolean;
}) {
  const [all, setAll] = useState(false);
  if (!ids.length) return null;
  const count = typeof total === "number" && total > ids.length ? total : ids.length;
  const partial = count > ids.length;
  const shown = all ? ids : ids.slice(0, TRIALS_SHOWN);
  const showAll = () => {
    onLoadAll?.();
    setAll(true);
  };
  return (
    <section className="trial-list" aria-label="Trials">
      <p className="section-label">{count.toLocaleString("en-US")} trial{count === 1 ? "" : "s"}</p>
      {partial && !unavailable && (all || ids.length <= TRIALS_SHOWN) && (
        <p className="muted small">Listing the first {ids.length.toLocaleString("en-US")}.</p>
      )}
      {partial && unavailable && <p className="muted small">Full trial list not available for this response.</p>}
      <ul className="nct-ids">
        {shown.map((id) => (
          <li key={id}>{NCT_RE.test(id)
            ? <a href={studyLink(id, studyUrl)} target="_blank" rel="noreferrer">{id}</a>
            : <span>{id}</span>}</li>
        ))}
      </ul>
      {partial && onLoadAll ? (
        // Only the first ids came inline: fetch the complete list (shared with the excerpt fetch).
        <button type="button" className="link-btn small" onClick={showAll} disabled={loading}>
          {loading ? "Loading…" : `Show all ${count.toLocaleString("en-US")}`}
        </button>
      ) : ids.length > TRIALS_SHOWN && (
        <button type="button" className="link-btn small" onClick={() => setAll(!all)}>
          {all ? "Show fewer" : `Show all ${ids.length.toLocaleString("en-US")}`}
        </button>
      )}
    </section>
  );
}

/** datum.query_url: the ClinicalTrials.gov search whose total is this value. */
function QueryLink({ url }: { url?: string }) {
  if (!url) return null;
  const href = safeHttpUrl(url);
  return (
    <p className="query-url">
      Count from ClinicalTrials.gov{href ? <> — <a href={href} target="_blank" rel="noreferrer">open this search</a></> : null}
    </p>
  );
}

interface TrialGroup { nct: string; items: Citation[] }

/** Consecutive excerpts from one trial share a card; repeated identical excerpts are shown once. */
function groupByTrial(cs: Citation[]): TrialGroup[] {
  const out: TrialGroup[] = [];
  for (const c of cs) {
    const nct = c.kind === "derived_count" ? "" : c.nct_id ?? "";
    const last = out[out.length - 1];
    if (last && nct && last.nct === nct) {
      if (!last.items.some((x) => x.excerpt === c.excerpt && x.label === c.label)) last.items.push(c);
    } else {
      out.push({ nct, items: [c] });
    }
  }
  return out;
}

/** One trial: its id (linked to the study page) and each excerpt's value, as plain text. */
function CitationItem({ group, studyUrl }: { group: TrialGroup; studyUrl?: string }) {
  const first = group.items[0];
  if (first.kind === "derived_count") {
    const href = safeHttpUrl(first.url);
    return (
      <li className="citation">
        Count from ClinicalTrials.gov{href ? <> — <a href={href} target="_blank" rel="noreferrer">open this search</a></> : null}
      </li>
    );
  }
  const nct = group.nct;
  const href = studyLink(nct, studyUrl) ?? safeHttpUrl(first.url);
  return (
    <li className="citation">
      {href
        ? <a href={href} target="_blank" rel="noreferrer" className="nct">{nct || "Trial record"}</a>
        : <span className="nct">{nct || "Trial record"}</span>}
      {group.items.map((c, i) => {
        const text = prettyEnum(c.excerpt);
        return (
          <p key={i} className="excerpt" title={text !== c.excerpt ? c.excerpt : undefined}>
            {c.label && <span className="citation-label">{c.label}: </span>}{text}
            {c.verified === false && (
              <span className="badge unverified" title="This excerpt no longer matches the stored trial record exactly.">not re-verified</span>
            )}
          </p>
        );
      })}
    </li>
  );
}
