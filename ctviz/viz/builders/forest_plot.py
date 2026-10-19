"""forest_plot: one row per effect estimate with CI; a reference line at the null value."""

from __future__ import annotations

from ctviz.api.schemas.viz import Channel, Datum, Encoding, ReferenceLine, Scale, Visualization, VizMeta, VizType
from ctviz.viz.base import make_viz


def forest_plot(
    rows: list[Datum],
    *,
    title: str,
    kind: str = "ratio",
    estimate_title: str = "Estimate",
    subtitle: str | None = None,
    meta: VizMeta | None = None,
    series: Channel | None = None,
    tooltip: list[str] | None = None,
) -> Visualization:
    """Rows {label, estimate, ci_low, ci_high, ci_pct, p_value, param_type}.

    ``kind='ratio'`` uses a log x scale with ``meta.reference_line`` at 1; ``'difference'`` a linear scale at 0.
    ``series`` colours rows by a field (e.g. the compared cohort when rows come from several trials).
    """
    meta = meta or VizMeta()
    meta.reference_line = ReferenceLine(value=1.0 if kind == "ratio" else 0.0, label="No difference")
    enc = Encoding(
        y=Channel(field="label", type="nominal", title="Comparison"),
        x=Channel(
            field="estimate",
            type="quantitative",
            title=estimate_title,
            scale=Scale(type="log" if kind == "ratio" else "linear"),
        ),
        error_low=Channel(field="ci_low", type="quantitative", title="CI lower"),
        error_high=Channel(field="ci_high", type="quantitative", title="CI upper"),
        series=series,
        tooltip=tooltip or ["p_value", "ci_pct", "param_type", "method"],
    )
    return make_viz(VizType.FOREST_PLOT, title, enc, rows, meta, subtitle)
