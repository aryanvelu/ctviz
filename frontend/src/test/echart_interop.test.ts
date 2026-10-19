// The production bundle once rendered `echarts-for-react/lib/core` as an object (React error #130):
// Vite's CJS interop hands over {default: Component} where vitest hands over the component. Not mocked here.
import { describe, expect, it } from "vitest";
import ReactEChartsCoreModule from "echarts-for-react/lib/core";
import { interopDefault, ReactEChartsCore } from "../viz/EChart";

describe("echarts-for-react interop", () => {
  it("the component EChart renders is a component (function / class), not a module object", () => {
    expect(typeof ReactEChartsCore).toBe("function");
  });

  it("unwraps a CJS namespace ({default: Component}) and passes a component through", () => {
    const Comp = ReactEChartsCoreModule;
    expect(interopDefault({ default: Comp, __esModule: true })).toBe(Comp);
    expect(interopDefault(Comp)).toBe(Comp);
  });
});
