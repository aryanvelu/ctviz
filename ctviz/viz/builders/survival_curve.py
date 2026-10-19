"""survival_curve: line per series of S(t). Rows {series, time, survival, source, ci_low?, ci_high?}."""

from __future__ import annotations

from ctviz.api.schemas.viz import Channel, Datum, Encoding, Scale, Visualization, VizMeta, VizType
from ctviz.viz.base import make_viz


def survival_curve(
    rows: list[Datum],
    *,
    title: str,
    time_unit: str = "months",
    subtitle: str | None = None,
    meta: VizMeta | None = None,
) -> Visualization:
    """Rows with ``source == 'interpolated'`` should be drawn dashed; the caller sets ``meta.approximation``."""
    meta = meta or VizMeta()
    meta.units.setdefault("time", time_unit)
    meta.units.setdefault("survival", "probability")
    enc = Encoding(
        x=Channel(
            field="time", type="quantitative", title=f"Time ({time_unit})", unit=time_unit, scale=Scale(zero=True)
        ),
        y=Channel(
            field="survival",
            type="quantitative",
            title="Survival probability",
            format=".0%",
            scale=Scale(domain=[0, 1]),
        ),
        series=Channel(field="series", type="nominal", title="Endpoint · arm"),
        tooltip=["source", "ci_low", "ci_high", "label"],
    )
    return make_viz(VizType.SURVIVAL_CURVE, title, enc, rows, meta, subtitle)
