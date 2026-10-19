// Query bar, persona toggle, structured filters, and patient profile.
import { useState, type FormEvent } from "react";
import { PHASES, STATUSES, VIZ_TYPES, type PatientProfile, type VisualizeRequest } from "../types";
import { humanize, prettyEnum } from "../format";

export type Persona = "auto" | "patient" | "researcher";

export interface FormState {
  query: string;
  persona: Persona;
  drug_name: string;
  condition: string;
  nct_id: string;
  trial_phase: string[];
  status: string[];
  sponsor: string;
  country: string;
  start_year: string;
  end_year: string;
  preferred_visualization: string;
  narrative: boolean;
  age: string;
  sex: string;
  conditions: string;
  medications: string;
}

export const EMPTY_FORM: FormState = {
  query: "", persona: "auto", drug_name: "", condition: "", nct_id: "", trial_phase: [], status: [],
  sponsor: "", country: "", start_year: "", end_year: "", preferred_visualization: "", narrative: true,
  age: "", sex: "", conditions: "", medications: "",
};

/** Quick status presets; each writes FormState.status (the More filters chips show the same array). */
export const STATUS_PRESETS: readonly { key: string; label: string; title: string; statuses: readonly string[] }[] = [
  { key: "active", label: "Active", title: "Active: recruiting, not yet recruiting, enrolling by invitation, active not recruiting",
    statuses: ["NOT_YET_RECRUITING", "RECRUITING", "ENROLLING_BY_INVITATION", "ACTIVE_NOT_RECRUITING"] },
  { key: "recruiting", label: "Recruiting", title: "Recruiting", statuses: ["RECRUITING"] },
  { key: "done", label: "Done", title: "Done: completed, terminated, withdrawn", statuses: ["COMPLETED", "TERMINATED", "WITHDRAWN"] },
  { key: "completed", label: "Completed", title: "Completed", statuses: ["COMPLETED"] },
  { key: "stopped", label: "Terminated or withdrawn", title: "Terminated or withdrawn", statuses: ["TERMINATED", "WITHDRAWN"] },
];

/** Which preset matches the current status list: "" = any status, "custom" = a hand-picked set from More filters. */
export function statusPresetKey(status: readonly string[]): string {
  if (!status.length) return "";
  const have = new Set(status);
  const hit = STATUS_PRESETS.find((p) => p.statuses.length === have.size && p.statuses.every((s) => have.has(s)));
  return hit ? hit.key : "custom";
}

export function statusesForPreset(key: string): string[] {
  return [...(STATUS_PRESETS.find((p) => p.key === key)?.statuses ?? [])];
}

const splitList = (s: string) => s.split(",").map((x) => x.trim()).filter(Boolean);

/** One value goes to the singular field, several (comma-separated) to the list field used for comparisons. */
function oneOrMany(s: string): { one?: string; many?: string[] } {
  const xs = splitList(s);
  return xs.length > 1 ? { many: xs } : { one: xs[0] };
}

export function toRequest(f: FormState): VisualizeRequest {
  const year = (s: string) => (s.trim() ? Number(s) : undefined);
  const drugs = oneOrMany(f.drug_name);
  const conds = oneOrMany(f.condition);
  const req: VisualizeRequest = {
    query: f.query.trim(),
    persona: f.persona,
    drug_name: drugs.one,
    drug_names: drugs.many,
    condition: conds.one,
    conditions: conds.many,
    nct_id: f.nct_id.trim().toUpperCase() || undefined,
    trial_phase: f.trial_phase,
    status: f.status,
    sponsor: f.sponsor.trim() || undefined,
    country: f.country.trim() || undefined,
    start_year: year(f.start_year),
    end_year: year(f.end_year),
    preferred_visualization: (f.preferred_visualization || undefined) as VisualizeRequest["preferred_visualization"],
    narrative: f.narrative,
  };
  if (f.persona === "patient") {
    const p: PatientProfile = {
      age: f.age.trim() ? Number(f.age) : undefined,
      sex: (f.sex || undefined) as PatientProfile["sex"],
      conditions: splitList(f.conditions),
      medications: splitList(f.medications),
    };
    req.patient_profile = p;
  }
  return req;
}

interface Props {
  form: FormState;
  setForm: (f: FormState) => void;
  onSubmit: (req: VisualizeRequest) => void;
  busy: boolean;
  onCancel: () => void;
}

function MultiSelect({ label, options, value, onChange }: { label: string; options: readonly string[]; value: string[]; onChange: (v: string[]) => void }) {
  return (
    <fieldset className="multi">
      <legend>{label}</legend>
      <div className="multi-opts">
        {options.map((o) => (
          <label key={o} className={`chip small ${value.includes(o) ? "active" : ""}`}>
            <input type="checkbox" checked={value.includes(o)} onChange={(e) => onChange(e.target.checked ? [...value, o] : value.filter((x) => x !== o))} />
            {prettyEnum(o)}
          </label>
        ))}
      </div>
    </fieldset>
  );
}

function StatusSelect({ value, onChange }: { value: string[]; onChange: (v: string[]) => void }) {
  const key = statusPresetKey(value);
  const current = STATUS_PRESETS.find((p) => p.key === key);
  return (
    <label>Status
      <select value={key} title={current?.title ?? (key === "custom" ? value.map(prettyEnum).join(", ") : "Any status")} onChange={(e) => onChange(statusesForPreset(e.target.value))}>
        <option value="" title="Any status (no filter)">Any status</option>
        {STATUS_PRESETS.map((p) => <option key={p.key} value={p.key} title={p.title}>{p.label}</option>)}
        {key === "custom" && <option value="custom" disabled>Custom (see More filters)</option>}
      </select>
    </label>
  );
}

export function QueryForm({ form, setForm, onSubmit, busy, onCancel }: Props) {
  const [filtersOpen, setFiltersOpen] = useState(false);
  const set = <K extends keyof FormState>(k: K, v: FormState[K]) => setForm({ ...form, [k]: v });
  const submit = (e?: Pick<FormEvent, "preventDefault">) => {
    e?.preventDefault();
    if (!form.query.trim() || busy) return;
    onSubmit(toRequest(form));
  };
  return (
    <form className="query-form" onSubmit={submit}>
      <div className="persona" role="radiogroup" aria-label="Persona">
        {(["auto", "patient", "researcher"] as Persona[]).map((p) => (
          <label key={p} className={`seg ${form.persona === p ? "active" : ""}`}>
            <input type="radio" name="persona" value={p} checked={form.persona === p} onChange={() => set("persona", p)} />
            {humanize(p)}
          </label>
        ))}
      </div>
      <div className="query-row">
        <label htmlFor="q" className="sr-only">Question</label>
        <textarea
          id="q"
          rows={2}
          placeholder="Ask about clinical trials (Enter to submit, Shift+Enter for a new line), e.g. “How many pembrolizumab trials started each year since 2015?”"
          value={form.query}
          maxLength={2000}
          onChange={(e) => set("query", e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) submit(e); }}
        />
        {busy
          ? <button type="button" className="secondary" onClick={onCancel}>Cancel</button>
          : <button type="submit" disabled={!form.query.trim()}>Visualize</button>}
      </div>
      <div className="quick-fields">
        <label>Drug(s)<input value={form.drug_name} onChange={(e) => set("drug_name", e.target.value)} placeholder="semaglutide, tirzepatide" /></label>
        <label>Condition(s)<input value={form.condition} onChange={(e) => set("condition", e.target.value)} placeholder="obesity" /></label>
        <label>NCT ID<input value={form.nct_id} onChange={(e) => set("nct_id", e.target.value)} placeholder="NCT03548935" pattern="^[Nn][Cc][Tt]\d{8}$" /></label>
        <StatusSelect value={form.status} onChange={(v) => set("status", v)} />
      </div>
      {form.persona === "patient" && (
        <fieldset className="profile">
          <legend>About you (optional, never stored)</legend>
          <div className="grid">
            <label>Age<input type="number" min={0} max={120} value={form.age} onChange={(e) => set("age", e.target.value)} /></label>
            <label>Sex
              <select value={form.sex} onChange={(e) => set("sex", e.target.value)}>
                <option value="">—</option><option value="female">Female</option><option value="male">Male</option><option value="other">Other</option>
              </select>
            </label>
            <label className="wide">Conditions (comma-separated)<input value={form.conditions} onChange={(e) => set("conditions", e.target.value)} placeholder="obesity, hypertension" /></label>
            <label className="wide">Medications (comma-separated)<input value={form.medications} onChange={(e) => set("medications", e.target.value)} placeholder="metformin" /></label>
          </div>
        </fieldset>
      )}
      <div className="filters">
        <button type="button" className="link-btn" aria-expanded={filtersOpen} aria-controls="filters-panel" onClick={() => setFiltersOpen(!filtersOpen)}>
          {filtersOpen ? "▾" : "▸"} More filters
        </button>
        {filtersOpen && (
          <div id="filters-panel" className="filters-panel">
            <div className="grid">
              <label>Sponsor<input value={form.sponsor} onChange={(e) => set("sponsor", e.target.value)} /></label>
              <label>Country<input value={form.country} onChange={(e) => set("country", e.target.value)} /></label>
              <label>Start year<input type="number" min={1990} value={form.start_year} onChange={(e) => set("start_year", e.target.value)} /></label>
              <label>End year<input type="number" min={1990} value={form.end_year} onChange={(e) => set("end_year", e.target.value)} /></label>
              <label>Preferred visualization
                <select value={form.preferred_visualization} onChange={(e) => set("preferred_visualization", e.target.value)}>
                  <option value="">Automatic</option>
                  {VIZ_TYPES.map((t) => <option key={t} value={t}>{humanize(t)}</option>)}
                </select>
              </label>
              <label className="check"><input type="checkbox" checked={form.narrative} onChange={(e) => set("narrative", e.target.checked)} /> Include summary</label>
            </div>
            <MultiSelect label="Phases" options={PHASES} value={form.trial_phase} onChange={(v) => set("trial_phase", v)} />
            <MultiSelect label="Status" options={STATUSES} value={form.status} onChange={(v) => set("status", v)} />
            <button type="button" className="link-btn small" onClick={() => setForm({ ...EMPTY_FORM, query: form.query, persona: form.persona })}>Clear filters</button>
          </div>
        )}
      </div>
    </form>
  );
}
