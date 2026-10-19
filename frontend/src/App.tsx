import { useEffect, useRef, useState } from "react";
import { ApiError, visualize, type VisualizeResult } from "./api";
import type { VisualizeRequest, VisualizeResponse, VizType } from "./types";
import { EMPTY_FORM, QueryForm, toRequest, type FormState } from "./components/QueryForm";
import { ResponseView } from "./components/ResponseView";

type Sample = { name: string; response: VisualizeResponse };

export default function App() {
  const [form, setForm] = useState<FormState>(EMPTY_FORM);
  const [result, setResult] = useState<VisualizeResult | null>(null);
  const [lastRequest, setLastRequest] = useState<VisualizeRequest | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [busy, setBusy] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const abort = useRef<AbortController | null>(null);

  useEffect(() => {
    if (!busy) return;
    const t0 = performance.now();
    setElapsed(0);
    const h = window.setInterval(() => setElapsed((performance.now() - t0) / 1000), 100);
    return () => window.clearInterval(h);
  }, [busy]);

  async function run(req: VisualizeRequest) {
    abort.current?.abort();
    const ctl = new AbortController();
    abort.current = ctl;
    setBusy(true);
    setError(null);
    setLastRequest(req);
    try {
      setResult(await visualize(req, ctl.signal));
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        setError(e instanceof ApiError ? e : new ApiError(0, "client_error", (e as Error).message));
      }
    } finally {
      if (abort.current === ctl) setBusy(false);
    }
  }

  function onAlternative(t: VizType) {
    const base = lastRequest ?? { query: form.query || "alternative view" };
    void run({ ...base, preferred_visualization: t });
  }

  /** Update the form (so the user sees what is asked) and submit it. */
  function ask(patch: Partial<FormState>) {
    const next = { ...form, ...patch };
    setForm(next);
    if (next.query.trim()) void run(toRequest(next));
  }

  const response = result?.data ?? null;

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>Clinical Trial Backend</h1>
          <p className="muted small">Questions about ClinicalTrials.gov, answered as cited charts.</p>
        </div>
        {import.meta.env.DEV && <SamplePicker onLoad={(s) => { setError(null); setResult({ data: s.response, raw: JSON.stringify(s.response) }); setLastRequest(null); }} />}
      </header>
      <QueryForm form={form} setForm={setForm} onSubmit={run} busy={busy} onCancel={() => { abort.current?.abort(); setBusy(false); }} />
      {busy && (
        <div className="loading" role="status" aria-live="polite">
          <span className="spinner" aria-hidden="true" /> Working… {elapsed.toFixed(1)} s
        </div>
      )}
      {error && <ErrorBox error={error} />}
      {result && (
        <ResponseView key={result.data.meta.request_id} response={result.data} raw={result.raw} busy={busy}
          onAlternative={onAlternative}
          onAsk={(q) => ask({ query: q })}
          onSuggest={(q) => { setForm({ ...form, query: q }); document.getElementById("q")?.focus(); }}
          onPickTrial={(id) => ask({ nct_id: id })} />
      )}
      {!response && !busy && !error && (
        <p className="muted empty">Ask a question above. Every bar, point and row links back to the trial record it came from.</p>
      )}
    </div>
  );
}

function ErrorBox({ error }: { error: ApiError }) {
  return (
    <div className="banner error" role="alert">
      <strong>{error.status === 429 ? "Rate limited" : error.status === 422 ? "Invalid request" : error.status === 503 ? "Service unavailable" : "Request failed"}</strong>
      {error.status ? <span className="muted small"> (HTTP {error.status}, {error.code})</span> : null}
      <p>{error.message}</p>
      {(error.status === 429 || error.status === 503) && error.retryAfterS !== undefined && <p>Try again in {Math.ceil(error.retryAfterS)} s.</p>}
      {error.issues.length > 0 && (
        <ul>
          {error.issues.map((i, k) => (
            <li key={k}><code>{i.loc.filter((l) => l !== "body").join(".") || "request"}</code>: {i.msg}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

/** Dev-only: render the bundled sample responses without a backend. Excluded from production builds. */
function SamplePicker({ onLoad }: { onLoad: (s: Sample) => void }) {
  const [samples, setSamples] = useState<Sample[]>([]);
  useEffect(() => {
    void import("./samples").then((m) => setSamples(m.SAMPLES));
  }, []);
  return (
    <label className="sample-picker small">
      Load sample{" "}
      <select defaultValue="" onChange={(e) => { const s = samples.find((x) => x.name === e.target.value); if (s) onLoad(s); }}>
        <option value="" disabled>— choose —</option>
        {samples.map((s) => <option key={s.name} value={s.name}>{s.name}</option>)}
      </select>
    </label>
  );
}
