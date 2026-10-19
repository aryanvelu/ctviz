"""geo_map, two levels (``meta.geo`` is a :class:`GeoSpec`):

- ``country``: one row per region; ``x`` = region name, ``y`` = value (so bar fallbacks just work);
  ``meta.geo.region_field`` / ``value_field`` join rows to shapes.
- ``site``: one row per site; ``label`` names the site and ``meta.geo.lat_field`` / ``lon_field`` place it.
  Rows without coordinates are listed but not drawn.
"""

from __future__ import annotations

from ctviz.api.schemas.viz import Channel, Datum, Encoding, GeoSpec, Visualization, VizMeta, VizType
from ctviz.viz.base import make_viz


def geo_map(
    rows: list[Datum],
    *,
    title: str,
    region: Channel,
    value: Channel,
    level: str = "country",
    series: Channel | None = None,
    subtitle: str | None = None,
    meta: VizMeta | None = None,
    tooltip: list[str] | None = None,
) -> Visualization:
    """Build a country-level (choropleth-style) map spec."""
    meta = meta or VizMeta()
    meta.units.setdefault(value.field, value.unit or "")
    meta.geo = GeoSpec(level=level, value_field=value.field, region_field=region.field)  # type: ignore[arg-type]
    enc = Encoding(x=region, y=value, series=series, label=region, tooltip=tooltip or [])
    return make_viz(VizType.GEO_MAP, title, enc, rows, meta, subtitle)


def site_map(
    rows: list[Datum],
    *,
    title: str,
    label: Channel,
    lat_field: str = "lat",
    lon_field: str = "lon",
    subtitle: str | None = None,
    meta: VizMeta | None = None,
    tooltip: list[str] | None = None,
) -> Visualization:
    """Build a site-level map spec: points at ``lat_field``/``lon_field`` (missing = not drawn)."""
    meta = meta or VizMeta()
    meta.geo = GeoSpec(level="site", lat_field=lat_field, lon_field=lon_field)
    enc = Encoding(label=label, tooltip=tooltip or [])
    return make_viz(VizType.GEO_MAP, title, enc, rows, meta, subtitle)
