import { afterEach } from "vitest";
import { cleanup } from "@testing-library/react";

// jsdom has no canvas; ECharts only needs text measurement during SSR layout.
HTMLCanvasElement.prototype.getContext = function getContext() {
  return { font: "", measureText: (t: string) => ({ width: t.length * 6 }) };
} as unknown as typeof HTMLCanvasElement.prototype.getContext;

afterEach(() => cleanup());
