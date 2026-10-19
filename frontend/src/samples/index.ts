// Sample public responses (schema 2.2, exactly what POST /v1/visualize returns), used by the
// dev-only "Load sample" dropdown and by the tests. live_*.json were captured from the running
// backend; the others are hand-written samples (one per visualization type) converted to 2.2 with
// the backend's own projection helpers (ctviz/api/public.py: _datum, citation_policy). trial_required.json adds candidate-trial options to a real
// clarification response.
import type { VisualizeResponse } from "../types";

const modules = import.meta.glob<VisualizeResponse>("./*.json", { eager: true, import: "default" });

export const SAMPLES: { name: string; response: VisualizeResponse }[] = Object.entries(modules)
  .map(([path, response]) => ({ name: path.replace("./", "").replace(".json", ""), response }))
  .sort((a, b) => a.name.localeCompare(b.name));
