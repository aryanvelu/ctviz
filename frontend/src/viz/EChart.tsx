// Thin wrapper around echarts-for-react using tree-shaken ECharts modules.
// Every series data item that represents a datum carries `datumId`; clicks resolve it.
import ReactEChartsCoreModule from "echarts-for-react/lib/core";
import * as echarts from "echarts/core";
import {
  BarChart, CustomChart, GraphChart, HeatmapChart, LineChart, PieChart, SankeyChart, ScatterChart,
} from "echarts/charts";
import {
  AriaComponent, DataZoomComponent, GridComponent, LegendComponent, MarkAreaComponent, MarkLineComponent,
  TooltipComponent, VisualMapComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import { useMemo } from "react";

/** CJS default interop: the production bundle hands us {default: Component}, vitest the component itself. */
export function interopDefault<T>(mod: T): T {
  const m = mod as unknown as { default?: unknown; __esModule?: boolean };
  return (typeof mod !== "function" && m && m.default ? m.default : mod) as T;
}

export const ReactEChartsCore = interopDefault(ReactEChartsCoreModule);

echarts.use([
  BarChart, CustomChart, GraphChart, HeatmapChart, LineChart, PieChart, SankeyChart, ScatterChart,
  AriaComponent, DataZoomComponent, GridComponent, LegendComponent, MarkAreaComponent, MarkLineComponent,
  TooltipComponent, VisualMapComponent, CanvasRenderer,
]);

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type EOption = Record<string, any>;

interface Props {
  option: EOption;
  height?: number;
  onSelect?: (datumId: string) => void;
  /** Short description; the accessible equivalent of the chart is the data table rendered below it. */
  ariaLabel: string;
}

export function EChart({ option, height = 420, onSelect, ariaLabel }: Props) {
  const onEvents = useMemo(() => ({
    click: (p: { data?: { datumId?: string } }) => {
      const id = p?.data?.datumId;
      if (id && onSelect) onSelect(id);
    },
  }), [onSelect]);
  // Same option object across re-renders, so echarts-for-react skips setOption (no redraw/re-animation).
  const full = useMemo(() => ({ aria: { enabled: true }, ...option }), [option]);
  const style = useMemo(() => ({ height, width: "100%" }), [height]);
  return (
    <div className="chart" aria-hidden="true" title={ariaLabel}>
      <ReactEChartsCore
        echarts={echarts}
        option={full}
        notMerge
        lazyUpdate
        style={style}
        onEvents={onEvents}
      />
    </div>
  );
}
