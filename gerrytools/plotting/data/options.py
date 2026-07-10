"""Public styling-options dataclasses for the data plots.

Each dataclass collects the styling kwargs that one ``add_*`` method takes,
so users can compose a style once and reuse it across calls. Kwargs on the
``add_*`` methods remain the primary path; ``*_options=`` is the secondary
compose-and-reuse path.

Resolution rule (documented per-method): start from ``options`` (or its
default if ``None``), then override each field with whatever explicit kwargs
the caller passed.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

from matplotlib.artist import Artist
from matplotlib.axes import Axes

from gerrytools.colors import resolve_color_and_alpha, resolve_rgba, validate_alpha
from gerrytools.logging import get_logger
from gerrytools.plotting.mpl.marker_options import PointMarkerOptions
from gerrytools.plotting.utils import (
    UNSET,
    Unset,
    _resolve_alpha_override,
    _resolve_color_clamped_width,
    _validated_nonneg_finite,
)
from gerrytools.typing import Color, HistType

logger = get_logger(__name__)


# Edge width applied when a visible edge color is set but no width is given. An edge
# color with zero width draws nothing, so asking for a color is taken to mean "draw the
# edge": the width falls back to this default rather than forcing the caller to set both.
DEFAULT_EDGE_WIDTH = 0.8


class _DefaultZorder(int):
    """Internal marker that survives ``dataclasses.replace``."""


_DEFAULT_ANNOTATION_ZORDER = _DefaultZorder(3)


def _resolve_annotation_zorder(value: int | float | Unset) -> tuple[int, bool]:
    if isinstance(value, Unset):
        return _DEFAULT_ANNOTATION_ZORDER, True
    if isinstance(value, _DefaultZorder):
        return value, True
    return int(value), False


def _needs_default_edge_width(
    *,
    edgewidth_given: bool,
    resolved_edgewidth: float,
    resolved_edgecolor: Color | None,
) -> bool:
    """Whether an unset edge width should fall back to ``DEFAULT_EDGE_WIDTH``.

    A visible edge color with zero width draws nothing, so naming an edge color while
    leaving the width unset is taken to mean "draw the edge". An explicit width of 0
    still hides it.
    """
    return (
        not edgewidth_given
        and resolved_edgewidth == 0.0
        and resolved_edgecolor is not None
        and str(resolved_edgecolor).strip().lower() != "none"
    )


# ---------------------------------------------------------------------------
# Lines and bands (used by GerryPlotBase add_*_lines / add_*_band).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LineOptions:
    """Styling for vertical/horizontal annotation lines.

    Attributes:
        linecolor (Color): The color of the line. Defaults to "#cccccc".
        linealpha (float | None): Optional alpha override.
        linestyle (str): Matplotlib linestyle. Defaults to "-".
        linewidth (float): Line width in points. Defaults to 1.0.
        zorder (int | float): Z-order for layering; coerced to int. Defaults to 3, but the
            annotation add methods substitute their documented orientation default
            (3 for vertical, 4 for horizontal) when this is left unset.
    """

    linecolor: Color = "#cccccc"
    linealpha: float | None = None
    linestyle: str = "-"
    linewidth: float = 1.0
    zorder: int | float | Unset = UNSET
    # True when the constructor received no explicit zorder; consumed (before any merge)
    # by the annotation add methods to substitute their orientation default.
    _zorder_defaulted: bool = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        owner = type(self).__name__
        line_width_value = _validated_nonneg_finite(self.linewidth, field="linewidth")
        resolved_linecolor, resolved_linealpha, line_width_value = _resolve_color_clamped_width(
            self.linecolor,
            self.linealpha,
            line_width_value,
            color_field="linecolor",
            width_field="linewidth",
            owner=owner,
        )
        object.__setattr__(self, "linecolor", resolved_linecolor)
        object.__setattr__(self, "linealpha", resolved_linealpha)
        object.__setattr__(self, "linewidth", line_width_value)
        zorder, defaulted = _resolve_annotation_zorder(self.zorder)
        object.__setattr__(self, "zorder", zorder)
        object.__setattr__(self, "_zorder_defaulted", defaulted)


@dataclass(frozen=True)
class BandOptions:
    """Styling for vertical/horizontal annotation bands (filled regions).

    Attributes:
        bandcolor (Color): The fill color of the band. Defaults to "#cccccc".
        bandalpha (float | None): Optional alpha override for the fill.
        linecolor (Color | None): Optional bounding-line color. ``None`` falls back to
            ``bandcolor`` (or ``"#cccccc"`` when the band fill is "none"), so the resolved
            value is always a concrete color.
        linealpha (float | None): Optional alpha override for the bounding lines.
        linestyle (str): Bounding-line linestyle. Defaults to "-".
        linewidth (float): Bounding-line width in points. Defaults to 1.0.
        zorder (int | float): Z-order for layering; coerced to int. Defaults to 3, but the
            annotation add methods substitute their documented orientation default
            (3 for vertical, 4 for horizontal) when this is left unset.
    """

    bandcolor: Color = "#cccccc"
    bandalpha: float | None = None
    linecolor: Color | None = None
    linealpha: float | None = None
    linestyle: str = "-"
    linewidth: float = 1.0
    zorder: int | float | Unset = UNSET
    # True when the constructor received no explicit zorder; consumed (before any merge)
    # by the annotation add methods to substitute their orientation default.
    _zorder_defaulted: bool = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        line_width_value = _validated_nonneg_finite(self.linewidth, field="linewidth")
        object.__setattr__(self, "linewidth", line_width_value)

        resolved_bandcolor, resolved_bandalpha = resolve_color_and_alpha(
            self.bandcolor,
            self.bandalpha,
            allow_none=True,
            field="bandcolor",
            owner="BandOptions",
            logger=logger,
        )
        object.__setattr__(self, "bandcolor", resolved_bandcolor)
        object.__setattr__(self, "bandalpha", resolved_bandalpha)

        # Bounding lines default to the band fill; a transparent fill falls back to the
        # neutral default so the band still has a visible boundary color.
        line_color_input = self.linecolor
        if line_color_input is None:
            line_color_input = resolved_bandcolor
            if isinstance(line_color_input, str) and line_color_input.lower() == "none":
                line_color_input = "#cccccc"
        resolved_linecolor, resolved_linealpha, line_width_value = _resolve_color_clamped_width(
            line_color_input,
            self.linealpha,
            line_width_value,
            color_field="linecolor",
            width_field="linewidth",
            owner="BandOptions",
        )
        object.__setattr__(self, "linecolor", resolved_linecolor)
        object.__setattr__(self, "linealpha", resolved_linealpha)
        object.__setattr__(self, "linewidth", line_width_value)
        zorder, defaulted = _resolve_annotation_zorder(self.zorder)
        object.__setattr__(self, "zorder", zorder)
        object.__setattr__(self, "_zorder_defaulted", defaulted)

    def resolved_edgecolor(
        self, *, owner: str = "BandOptions"
    ) -> str | tuple[float, float, float, float]:
        """Edge color to draw the band's bounding lines with.

        Encodes the one shared drawing rule: zero-width bounding lines resolve to
        ``"none"`` so matplotlib's default hairline edge never appears.
        """
        if self.linewidth == 0.0:
            return "none"
        return resolve_rgba(self.linecolor, self.linealpha, field="linecolor", owner=owner)


# ---------------------------------------------------------------------------
# Histogram, BarPlot, BoxPlot, ViolinPlot.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FaceEdgeStyle:
    """Shared face/edge styling block with one validation pass.

    The four distribution-family options classes inherit this: resolved colors, the
    edge-width checks, the invisible-edge clamp, and zorder coercion live here once.

    Attributes:
        facecolor (Color): Fill color.
        facealpha (float | None): Optional alpha override for the fill.
        edgecolor (Color): Edge color.
        edgealpha (float | None): Optional alpha override for the edge.
        edgewidth (float): Edge line width in points.
        zorder (int | float): Z-order for layering; coerced to int.
    """

    facecolor: Color = "default_grey"
    facealpha: float | None = None
    edgecolor: Color = "black"
    edgealpha: float | None = None
    edgewidth: float = 0.8
    zorder: int | float = 1

    def __post_init__(self) -> None:
        owner = type(self).__name__

        edge_width_value = _validated_nonneg_finite(self.edgewidth, field="edgewidth")
        object.__setattr__(self, "edgewidth", edge_width_value)

        resolved_facecolor, resolved_facealpha = resolve_color_and_alpha(
            self.facecolor,
            self.facealpha,
            allow_none=True,
            field="facecolor",
            owner=owner,
            logger=logger,
        )
        object.__setattr__(self, "facecolor", resolved_facecolor)
        object.__setattr__(self, "facealpha", resolved_facealpha)

        resolved_edgecolor, resolved_edgealpha, edge_width_value = _resolve_color_clamped_width(
            self.edgecolor,
            self.edgealpha,
            edge_width_value,
            color_field="edgecolor",
            width_field="edgewidth",
            owner=owner,
        )
        object.__setattr__(self, "edgecolor", resolved_edgecolor)
        object.__setattr__(self, "edgealpha", resolved_edgealpha)
        object.__setattr__(self, "edgewidth", edge_width_value)

        object.__setattr__(self, "zorder", int(self.zorder))

    def merged(
        self,
        *,
        facecolor: Color | None | Unset = UNSET,
        facealpha: float | None = None,
        edgecolor: Color | None | Unset = UNSET,
        edgealpha: float | None = None,
        **other: Any,
    ) -> Any:
        """Copy this style with the caller's explicit overrides applied.

        Encodes the color/alpha pairing rule once: an explicit alpha always wins; overriding
        only a color keeps the base alpha unless the base color was the fully transparent
        "none", whose 0.0 alpha would render the override invisibly. Colors use the ``UNSET``
        sentinel so an explicit ``None`` means "none" while an omitted kwarg inherits; every
        other field treats ``None`` as "inherit". The merged result re-runs validation.

        Use this for the face/edge options classes; ``_replace_non_none`` in
        :mod:`gerrytools.plotting.utils` is the plain None-inherits merge for every other
        options dataclass, where ``None`` is never a meaningful field value.

        Returns:
            A new instance of the same options class (``self`` when nothing was overridden).
        """
        updates: dict[str, Any] = {key: value for key, value in other.items() if value is not None}
        if facecolor is not UNSET or facealpha is not None:
            updates["facecolor"] = self.facecolor if facecolor is UNSET else facecolor
            updates["facealpha"] = _resolve_alpha_override(
                facecolor is not UNSET, facealpha, self.facecolor, self.facealpha
            )
        if edgecolor is not UNSET or edgealpha is not None:
            updates["edgecolor"] = self.edgecolor if edgecolor is UNSET else edgecolor
            updates["edgealpha"] = _resolve_alpha_override(
                edgecolor is not UNSET, edgealpha, self.edgecolor, self.edgealpha
            )
        if not updates:
            return self
        return dataclasses.replace(self, **updates)


@dataclass(frozen=True)
class HistogramOptions(_FaceEdgeStyle):
    """Styling for a single histogram series added via ``Histogram.add_dataset``.

    Defaults mirror the previous ``add_dataset`` kwargs: a filled bar with no
    visible edge. For ``histtype="outline"`` the method itself enforces the
    sensible-outline overrides (positive ``edgewidth``, ``facecolor="none"``,
    ``edgecolor="black"``).

    Attributes:
        facecolor (Color): Fill color for histogram bars.
        facealpha (float | None): Optional alpha override for the fill.
        edgecolor (Color): Edge color for histogram bars.
        edgealpha (float | None): Optional alpha override for the edge.
        edgewidth (float): Edge line width in points.
        histtype (HistType): One of "overlay", "stack", "grouped", "outline".
        zorder (int): Z-order for layering.
    """

    edgecolor: Color = "none"
    edgewidth: float = 0.0
    histtype: HistType = "overlay"
    zorder: int = 2


@dataclass(frozen=True)
class BarPlotOptions(_FaceEdgeStyle):
    """Styling for a single bar dataset added via ``BarPlot.add_dataset`` or
    ``BarPlot.add_counts_dataset``.

    Attributes:
        facecolor (Color): Fill color for bars.
        facealpha (float | None): Optional alpha override for the fill.
        edgecolor (Color): Edge color for bars.
        edgealpha (float | None): Optional alpha override for the edge.
        edgewidth (float): Edge line width.
        zorder (int): Z-order for layering.
    """


@dataclass(frozen=True)
class BoxPlotOptions(_FaceEdgeStyle):
    """Styling for a single boxplot dataset added via ``BoxPlot.add_dataset``.

    Attributes:
        facecolor (Color): Fill color for boxes.
        facealpha (float | None): Optional alpha override for the fill.
        edgecolor (Color): Edge color for boxes/whiskers.
        edgealpha (float | None): Optional alpha override for the edge.
        edgewidth (float): Edge line width.
        percentiles (tuple[float, float]): Whisker percentile bounds; both values must
            be in ``[0, 100]`` and ``low < high``.
        showfliers (bool): Whether to render outlier points.
        flier_options (PointMarkerOptions): Marker styling for fliers (outliers).
        zorder (int): Z-order for layering.
    """

    percentiles: tuple[float, float] = (1, 99)
    showfliers: bool = False
    flier_options: PointMarkerOptions = field(default_factory=PointMarkerOptions)

    def __post_init__(self) -> None:
        percentile_low, percentile_high = self.percentiles
        percentile_low = float(percentile_low)
        percentile_high = float(percentile_high)
        if not (0.0 <= percentile_low <= 100.0 and 0.0 <= percentile_high <= 100.0):
            raise ValueError("percentiles must be within [0, 100].")
        if not (percentile_low < percentile_high):
            raise ValueError("percentiles must satisfy low < high.")
        object.__setattr__(self, "percentiles", (percentile_low, percentile_high))

        super().__post_init__()


@dataclass(frozen=True)
class ViolinPlotOptions(_FaceEdgeStyle):
    """Styling for a single violin dataset added via ``ViolinPlot.add_dataset``.

    Attributes:
        facecolor (Color): Fill color for violins.
        facealpha (float | None): Optional alpha override for the fill.
        edgecolor (Color): Edge color for violin outline.
        edgealpha (float | None): Optional alpha override for the edge.
        edgewidth (float): Edge line width.
        zorder (int): Z-order for layering.
    """


# ---------------------------------------------------------------------------
# SeatsVotesPlot — line and marker subsets.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeatsVotesLineOptions:
    """Styling for the seats-votes curve line in ``SeatsVotesPlot.add_election``.

    Fields ``linewidth`` and ``linealpha`` may be ``None`` to inherit from
    plot-level defaults.

    Attributes:
        linecolor (Color | None): Curve color; ``None`` inherits from caller.
        linealpha (float | None): Optional alpha override.
        linestyle (str): Matplotlib linestyle. Defaults to "-".
        linewidth (float | None): Optional width override; ``None`` inherits.
        zorder (int | float): Z-order for the curve; coerced to int.
    """

    linecolor: Color | None = None
    linealpha: float | None = None
    linestyle: str = "-"
    linewidth: float | None = None
    zorder: int | float = 1

    def __post_init__(self) -> None:
        if self.linealpha is not None:
            object.__setattr__(self, "linealpha", validate_alpha(self.linealpha, field="linealpha"))

        if self.linewidth is not None:
            object.__setattr__(
                self, "linewidth", _validated_nonneg_finite(self.linewidth, field="linewidth")
            )

        if self.linecolor is not None:
            resolved_linecolor, resolved_linealpha = resolve_color_and_alpha(
                self.linecolor,
                self.linealpha,
                allow_none=True,
                field="linecolor",
                owner="SeatsVotesLineOptions",
                logger=logger,
            )
            object.__setattr__(self, "linecolor", resolved_linecolor)
            object.__setattr__(self, "linealpha", resolved_linealpha)

        object.__setattr__(self, "zorder", int(self.zorder))


@dataclass(frozen=True)
class SeatsVotesMarkerOptions:
    """Styling for the election-result marker in ``SeatsVotesPlot.add_election``.

    Attributes:
        markerfacecolor (Color | None): Marker fill color; ``None`` inherits from
            caller-provided color.
        markerfacealpha (float | None): Optional alpha override for the fill.
        marker (str): Matplotlib marker style. Defaults to "o".
        markersize (float | None): Optional size override; ``None`` inherits.
        markeredgecolor (Color | None): Optional edge color; ``None`` falls back
            to the marker face color at render time.
        markeredgealpha (float | None): Optional alpha override for the edge.
        markeredgewidth (float): Marker edge width.
        marker_zorder (int | float): Z-order for the marker; coerced to int.
    """

    markerfacecolor: Color | None = None
    markerfacealpha: float | None = None
    marker: str = "o"
    markersize: float | None = None
    markeredgecolor: Color | None = None
    markeredgealpha: float | None = None
    markeredgewidth: float = 0.0
    marker_zorder: int | float = 2

    def __post_init__(self) -> None:
        if self.markerfacealpha is not None:
            object.__setattr__(
                self,
                "markerfacealpha",
                validate_alpha(self.markerfacealpha, field="markerfacealpha"),
            )

        if self.markersize is not None:
            object.__setattr__(
                self, "markersize", _validated_nonneg_finite(self.markersize, field="markersize")
            )

        if self.markeredgealpha is not None:
            object.__setattr__(
                self,
                "markeredgealpha",
                validate_alpha(self.markeredgealpha, field="markeredgealpha"),
            )

        object.__setattr__(
            self,
            "markeredgewidth",
            _validated_nonneg_finite(self.markeredgewidth, field="markeredgewidth"),
        )

        if self.markerfacecolor is not None:
            resolved_face, resolved_face_alpha = resolve_color_and_alpha(
                self.markerfacecolor,
                self.markerfacealpha,
                allow_none=True,
                field="markerfacecolor",
                owner="SeatsVotesMarkerOptions",
                logger=logger,
            )
            object.__setattr__(self, "markerfacecolor", resolved_face)
            object.__setattr__(self, "markerfacealpha", resolved_face_alpha)

        if self.markeredgecolor is not None:
            resolved_edge, resolved_edge_alpha = resolve_color_and_alpha(
                self.markeredgecolor,
                self.markeredgealpha,
                allow_none=True,
                field="markeredgecolor",
                owner="SeatsVotesMarkerOptions",
                logger=logger,
            )
            object.__setattr__(self, "markeredgecolor", resolved_edge)
            object.__setattr__(self, "markeredgealpha", resolved_edge_alpha)

        object.__setattr__(self, "marker_zorder", int(self.marker_zorder))


# ---------------------------------------------------------------------------
# SeaLevelPlot — line subset only (markers reuse PointMarkerOptions).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeaLevelLineOptions(LineOptions):
    """Styling for the connecting line in ``SeaLevelPlot.add_dataset``.

    A ``LineOptions`` overriding only the defaults: a black, slightly heavier line drawn
    above the sea-level markers.

    Attributes:
        linecolor (Color): Line color. Defaults to "black".
        linealpha (float | None): Optional alpha override.
        linestyle (str): Matplotlib linestyle.
        linewidth (float): Line width in points. Defaults to 1.5.
        zorder (int | float): Z-order for the line; coerced to int. Defaults to 2.
    """

    linecolor: Color = "black"
    linewidth: float = 1.5
    zorder: int | float = 2


@dataclass(frozen=True, slots=True)
class _CrosshairStyle:
    """Center crosshair guides at (0.5, 0.5), shared by SeatsVotesPlot and PaintballPlot.

    Widths are data-space band widths; the color resolves at draw time.

    Attributes:
        color (Color): Crosshair color. Defaults to "lightgrey".
        alpha (float): Crosshair alpha in [0, 1]. Defaults to 1.0.
        x_width (float): Width of the vertical band in data units. Defaults to 0.02.
        y_width (float): Width of the horizontal band in data units. Defaults to 0.02.
        zorder (int): Draw order. Defaults to -2.
    """

    color: Color = "lightgrey"
    alpha: float = 1.0
    x_width: float = 0.02
    y_width: float = 0.02
    zorder: int = -2

    def __post_init__(self) -> None:
        object.__setattr__(self, "alpha", validate_alpha(self.alpha, field="alpha"))
        object.__setattr__(self, "x_width", _validated_nonneg_finite(self.x_width, field="x_width"))
        object.__setattr__(self, "y_width", _validated_nonneg_finite(self.y_width, field="y_width"))

    def draw(self, ax: Axes) -> list[Artist]:
        """Draw the two crosshair spans onto ``ax`` and return the created artists."""
        color = resolve_rgba(
            self.color, self.alpha, field="crosshair_color", owner="_CrosshairStyle"
        )
        vspan = ax.axvspan(
            xmin=0.5 - self.x_width / 2,
            xmax=0.5 + self.x_width / 2,
            color=color,
            zorder=self.zorder,
        )
        hspan = ax.axhspan(
            ymin=0.5 - self.y_width / 2,
            ymax=0.5 + self.y_width / 2,
            color=color,
            zorder=self.zorder,
        )
        return [vspan, hspan]


@dataclass(frozen=True, slots=True)
class _PaintballHullStyle:
    """Horizontal-hull styling for ``PaintballPlot``; None colors inherit the marker style.

    Attributes:
        facecolor (Color | None): Hull fill color; None inherits the marker face color.
        facealpha (float | None): Hull fill alpha; None inherits the marker face alpha.
        edgecolor (Color | None): Hull edge color; None inherits the marker edge color.
        edgealpha (float | None): Hull edge alpha; None inherits the marker edge alpha.
        edgewidth (float): Hull edge width in points. Defaults to 2.0.
    """

    facecolor: Color | None = None
    facealpha: float | None = None
    edgecolor: Color | None = None
    edgealpha: float | None = None
    edgewidth: float = 2.0

    def __post_init__(self) -> None:
        if self.facealpha is not None:
            object.__setattr__(self, "facealpha", validate_alpha(self.facealpha, field="alpha"))
        if self.edgealpha is not None:
            object.__setattr__(self, "edgealpha", validate_alpha(self.edgealpha, field="edgealpha"))
        object.__setattr__(
            self, "edgewidth", _validated_nonneg_finite(self.edgewidth, field="edgewidth")
        )
