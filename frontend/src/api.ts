// HTTP client for the Clinical Trial Backend API. Relative URLs: Vite proxies /v1 in dev, FastAPI serves both in prod.
import type { Citation, VisualizeRequest, VisualizeResponse } from "./types";

export interface ValidationIssue { loc: (string | number)[]; msg: string; type?: string }

export class ApiError extends Error {
  status: number;
  code: string;
  retryAfterS?: number;
  issues: ValidationIssue[];

  constructor(status: number, code: string, message: string, opts: { retryAfterS?: number; issues?: ValidationIssue[] } = {}) {
    super(message);
    this.status = status;
    this.code = code;
    this.retryAfterS = opts.retryAfterS;
    this.issues = opts.issues ?? [];
  }
}

async function toApiError(res: Response): Promise<ApiError> {
  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    // non-JSON error body (e.g. proxy error page)
  }
  const header = res.headers.get("Retry-After");
  const b = (body ?? {}) as Record<string, unknown>;
  const retry = typeof b.retry_after_s === "number" ? b.retry_after_s : header ? Number(header) : undefined;
  if (res.status === 422 && Array.isArray(b.detail)) {
    return new ApiError(422, "validation_error", "The request did not pass validation.", { issues: b.detail as ValidationIssue[] });
  }
  const code = typeof b.error === "string" ? b.error
    : res.status === 429 ? "rate_limited" : res.status === 503 ? "unavailable" : `http_${res.status}`;
  const message = typeof b.message === "string" ? b.message
    : typeof b.detail === "string" ? b.detail
      : res.status === 429 ? "Too many requests."
        : res.status === 503 ? "The service is temporarily unavailable." : res.statusText || "Request failed.";
  return new ApiError(res.status, code, message, { retryAfterS: Number.isFinite(retry) ? retry : undefined });
}

/** Drop empty strings/arrays so the server's structured-field overrides only see real input. */
export function cleanRequest(req: VisualizeRequest): VisualizeRequest {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(req)) {
    if (v === undefined || v === null || v === "") continue;
    if (Array.isArray(v) && v.length === 0) continue;
    if (k === "patient_profile" && typeof v === "object") {
      const p = cleanRequest(v as VisualizeRequest) as unknown as Record<string, unknown>;
      if (Object.keys(p).length) out[k] = p;
      continue;
    }
    out[k] = v;
  }
  return out as unknown as VisualizeRequest;
}

export interface VisualizeResult {
  data: VisualizeResponse;
  /** The response body exactly as received (shown verbatim in the "Response JSON" panel). */
  raw: string;
}

export async function visualize(req: VisualizeRequest, signal?: AbortSignal): Promise<VisualizeResult> {
  let res: Response;
  try {
    res = await fetch("/v1/visualize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(cleanRequest(req)),
      signal,
    });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ApiError(0, "network_error", "Could not reach the API. Is the backend running on port 8000?");
  }
  if (!res.ok) throw await toApiError(res);
  const raw = await res.text();
  let data: VisualizeResponse;
  try {
    data = JSON.parse(raw) as VisualizeResponse;
  } catch {
    throw new ApiError(res.status, "bad_json", "The API returned a response that is not JSON.");
  }
  if (!data || typeof data !== "object" || !data.meta) {
    throw new ApiError(res.status, "bad_shape", "The API response has no meta block; is the backend on schema 2.0?");
  }
  return { data, raw };
}

/** Everything the all-citations endpoint returns for one datum. */
export interface DatumSources {
  citations: Citation[];
  /** Schema 2.2+: the complete, sorted list of trials behind the datum. */
  nct_ids?: string[];
  nct_ids_total?: number;
}

/**
 * Full citation list (and, from schema 2.2, the complete nct_ids) for a truncated datum.
 * Accepts `{citations: [...], nct_ids?, nct_ids_total?}` or a bare citation array.
 */
export async function fetchAllCitations(requestId: string, datumId: string, signal?: AbortSignal): Promise<DatumSources> {
  const res = await fetch(`/v1/responses/${encodeURIComponent(requestId)}/datums/${encodeURIComponent(datumId)}/citations`, { signal });
  if (!res.ok) throw await toApiError(res);
  const body = (await res.json()) as unknown;
  if (Array.isArray(body)) return { citations: body as Citation[] };
  const b = (body ?? {}) as { citations?: unknown; nct_ids?: unknown; nct_ids_total?: unknown };
  const out: DatumSources = { citations: Array.isArray(b.citations) ? (b.citations as Citation[]) : [] };
  if (Array.isArray(b.nct_ids)) out.nct_ids = b.nct_ids.filter((x): x is string => typeof x === "string");
  if (typeof b.nct_ids_total === "number") out.nct_ids_total = b.nct_ids_total;
  return out;
}
