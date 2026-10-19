// App-level behavior: request building, success rendering, 422 and 429 error display.
import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import App from "../App";
import { SAMPLES } from "../samples";
import { cleanRequest } from "../api";
import { EMPTY_FORM, toRequest } from "../components/QueryForm";

vi.mock("echarts-for-react/lib/core", () => ({ default: () => <div data-testid="echart" /> }));

afterEach(() => vi.unstubAllGlobals());

function mockFetch(status: number, body: unknown, headers: Record<string, string> = {}) {
  const fn = vi.fn(async () => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json", ...headers } }));
  vi.stubGlobal("fetch", fn);
  return fn;
}

async function ask(text: string) {
  fireEvent.change(screen.getByLabelText("Question"), { target: { value: text } });
  fireEvent.click(screen.getByRole("button", { name: "Visualize" }));
}

describe("App", () => {
  it("builds a clean request with structured fields and patient profile", () => {
    const req = cleanRequest(toRequest({ ...EMPTY_FORM, query: "Am I eligible?", persona: "patient", nct_id: "nct03548935", age: "45", conditions: "obesity, asthma", trial_phase: ["PHASE3"] }));
    expect(req).toEqual({
      query: "Am I eligible?", persona: "patient", nct_id: "NCT03548935", trial_phase: ["PHASE3"], narrative: true,
      patient_profile: { age: 45, conditions: ["obesity", "asthma"] },
    });
  });

  it("posts to /v1/visualize and renders the response", async () => {
    const sample = SAMPLES.find((s) => s.name === "time_series")!.response;
    const fetchMock = mockFetch(200, sample);
    render(<App />);
    await ask("pembrolizumab trials per year");
    await waitFor(() => expect(screen.getByRole("heading", { name: sample.visualization!.title })).toBeTruthy());
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("/v1/visualize");
    expect(JSON.parse(String(init.body)).query).toBe("pembrolizumab trials per year");
    expect(screen.getByRole("region", { name: "Answer" })).toBeTruthy();
  });

  it("shows 422 validation details", async () => {
    mockFetch(422, { detail: [{ loc: ["body", "nct_id"], msg: "String should match pattern '^NCT\\d{8}$'", type: "string_pattern_mismatch" }] });
    render(<App />);
    await ask("bad");
    await waitFor(() => expect(screen.getByText("Invalid request")).toBeTruthy());
    expect(screen.getByText("nct_id")).toBeTruthy();
  });

  it("shows 429 with Retry-After", async () => {
    mockFetch(429, { error: "rate_limited", message: "Too many requests", retry_after_s: 12 }, { "Retry-After": "12" });
    render(<App />);
    await ask("anything");
    await waitFor(() => expect(screen.getByText("Rate limited")).toBeTruthy());
    expect(screen.getByText("Try again in 12 s.")).toBeTruthy();
  });
});
