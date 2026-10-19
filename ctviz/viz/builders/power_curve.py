"""power_curve: power vs standardized effect size, one series per sample-size scenario."""

from __future__ import annotations

from ctviz.api.schemas.viz import Channel, Datum, Encoding, Scale, Threshold, Visualization, VizMeta, VizType
from ctviz.viz.base import make_viz


def power_curve(
    rows: list[Datum], *, title: str, target: float = 0.8, subtitle: str | None = None, meta: VizMeta | None = None
) -> Visualization:
    """Rows {series, effect_size, power, n_per_arm}; adds the target-power threshold (convention)."""
    meta = meta or VizMeta()
    if not any(t.axis == "y" and t.value == target for t in meta.thresholds):
        meta.thresholds.append(
            Threshold(axis="y", value=target, label=f"{target:.0%} power (convention)", source="convention")
        )
    meta.units.setdefault("effect_size", "Cohen's d")
    meta.units.setdefault("power", "probability")
    enc = Encoding(
        x=Channel(
            field="effect_size",
            type="quantitative",
            title="Standardized effect size (d)",
            unit="Cohen's d",
            format=".2f",
        ),
        y=Channel(field="power", type="quantitative", title="Power", format=".0%", scale=Scale(domain=[0, 1])),
        series=Channel(field="series", type="nominal", title="Sample size scenario"),
        tooltip=["n_per_arm"],
    )
    return make_viz(VizType.POWER_CURVE, title, enc, rows, meta, subtitle)
