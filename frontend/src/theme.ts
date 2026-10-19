// Color tokens for charts. CSS uses the same values via custom properties in styles.css.
import { useEffect, useState } from "react";

export interface ChartTheme {
  dark: boolean;
  text: string;
  muted: string;
  grid: string;
  bg: string;
  surface: string;
  palette: string[];
  sequential: [string, string, string];
  gap: string;
  threshold: string;
  band: string;
  reference: string;
}

// Categorical palette (Observable 10 ordering) chosen for contrast on both light and dark surfaces.
const PALETTE_LIGHT = ["#4269d0", "#d9822b", "#c2410c", "#2f9e8f", "#3ca951", "#c2185b", "#7c3aed", "#5c7cfa", "#8d6e63", "#6b7280"];
const PALETTE_DARK = ["#7aa2ff", "#f2b84b", "#ff8a65", "#5fd4c0", "#6ed37f", "#ff7eb6", "#b794f6", "#a5b4fc", "#c4a484", "#a1a1aa"];

export const LIGHT: ChartTheme = {
  dark: false,
  text: "#1f2937",
  muted: "#4b5563",
  grid: "#e5e7eb",
  bg: "#ffffff",
  surface: "#f8fafc",
  palette: PALETTE_LIGHT,
  sequential: ["#eef2ff", "#7c9cf0", "#1e3a8a"],
  gap: "#c2410c",
  threshold: "#b91c1c",
  band: "rgba(66,105,208,0.10)",
  reference: "#374151",
};

export const DARK: ChartTheme = {
  dark: true,
  text: "#e5e7eb",
  muted: "#9ca3af",
  grid: "#374151",
  bg: "#111827",
  surface: "#1f2937",
  palette: PALETTE_DARK,
  sequential: ["#1e293b", "#3b5bb5", "#a5c0ff"],
  gap: "#ff8a65",
  threshold: "#f87171",
  band: "rgba(122,162,255,0.14)",
  reference: "#d1d5db",
};

function prefersDark(): boolean {
  return typeof window !== "undefined" && typeof window.matchMedia === "function"
    && window.matchMedia("(prefers-color-scheme: dark)").matches;
}

/** Follows the OS color scheme and re-renders charts when it changes. */
export function useChartTheme(): ChartTheme {
  const [dark, setDark] = useState(prefersDark);
  useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") return;
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const on = (e: MediaQueryListEvent) => setDark(e.matches);
    mq.addEventListener?.("change", on);
    return () => mq.removeEventListener?.("change", on);
  }, []);
  return dark ? DARK : LIGHT;
}

/** Stable color per category key, so the same series keeps its color across charts. */
export function colorFor(theme: ChartTheme, index: number): string {
  return theme.palette[index % theme.palette.length];
}

/** Base ECharts option pieces shared by every cartesian chart. */
export function baseOption(theme: ChartTheme) {
  return {
    backgroundColor: "transparent",
    color: theme.palette,
    textStyle: { color: theme.text, fontFamily: "inherit" },
    animationDuration: 300,
    tooltip: {
      confine: true,
      backgroundColor: theme.surface,
      borderColor: theme.grid,
      textStyle: { color: theme.text },
    },
    legend: { textStyle: { color: theme.text }, type: "scroll" as const, top: 0 },
  };
}

export function axisStyle(theme: ChartTheme) {
  return {
    axisLine: { lineStyle: { color: theme.muted } },
    axisLabel: { color: theme.muted },
    splitLine: { lineStyle: { color: theme.grid } },
    nameTextStyle: { color: theme.muted },
  };
}
