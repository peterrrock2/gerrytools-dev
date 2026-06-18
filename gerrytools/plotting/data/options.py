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

import logging
import math
from dataclasses import dataclass, field

from gerrytools.colors import resolve_color_and_alpha
from gerrytools.logging import get_logger
from gerrytools.plotting.mpl.marker_options import PointMarkerOptions
from gerrytools.typing import Color, HistType

logger = get_logger(__name__)


# Edge width applied when a visible edge color is set but no width is given. An edge
# color with zero width draws nothing, so asking for a color is taken to mean "draw the
# edge": the width falls back to this default rather than forcing the caller to set both.
DEFAULT_EDGE_WIDTH = 0.8


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
        zorder (int): Z-order for layering. Defaults to 3.
    """

    linecolor: Color = "#cccccc"
    linealpha: float | None = None
    linestyle: str = "-"
    linewidth: float = 1.0
    zorder: int = 3

    def __post_init__(self) -> None:
        line_width_value = float(self.linewidth)
        if not math.isfinite(line_width_value):
            raise ValueError("linewidth must be finite.")
        if line_width_value < 0:
            raise ValueError("linewidth must be nonnegative.")
        object.__setattr__(self, "linewidth", line_width_value)

        resolved_linecolor, resolved_linealpha = resolve_color_and_alpha(
            self.linecolor,
            self.linealpha,
            allow_none=True,
            field="linecolor",
            owner="LineOptions",
            logger=logger,
        )
        object.__setattr__(self, "linecolor", resolved_linecolor)
        object.__setattr__(self, "linealpha", resolved_linealpha)

        if resolved_linecolor.lower() == "none" and line_width_value > 0:
            logger.log(
                level=logging.DEBUG,
                msg=(
                    "LineOptions: linecolor is 'none' but "
                    f"linewidth is {line_width_value}>0; setting linewidth to 0."
                ),
            )
            object.__setattr__(self, "linewidth", 0.0)

        object.__setattr__(self, "zorder", int(self.zorder))


@dataclass(frozen=True)
class BandOptions:
    """Styling for vertical/horizontal annotation bands (filled regions).

    Attributes:
        bandcolor (Color): The fill color of the band. Defaults to "#cccccc".
        bandalpha (float | None): Optional alpha override for the fill.
        linecolor (Color | None): Optional bounding-line color. If ``None``, falls back
            to ``bandcolor``.
        linealpha (float | None): Optional alpha override for the bounding lines.
        linestyle (str): Bounding-line linestyle. Defaults to "-".
        linewidth (float): Bounding-line width in points. Defaults to 1.0.
        zorder (int): Z-order for layering. Defaults to 3.
    """

    bandcolor: Color = "#cccccc"
    bandalpha: float | None = None
    linecolor: Color | None = None
    linealpha: float | None = None
    linestyle: str = "-"
    linewidth: float = 1.0
    zorder: int = 3

    def __post_init__(self) -> None:
        line_width_value = float(self.linewidth)
        if not math.isfinite(line_width_value):
            raise ValueError("linewidth must be finite.")
        if line_width_value < 0:
            raise ValueError("linewidth must be nonnegative.")
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

        if self.linecolor is not None:
            resolved_linecolor, resolved_linealpha = resolve_color_and_alpha(
                self.linecolor,
                self.linealpha,
                allow_none=True,
                field="linecolor",
                owner="BandOptions",
                logger=logger,
            )
            object.__setattr__(self, "linecolor", resolved_linecolor)
            object.__setattr__(self, "linealpha", resolved_linealpha)

        object.__setattr__(self, "zorder", int(self.zorder))


# ---------------------------------------------------------------------------
# Histogram, BoxPlot, ViolinPlot.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HistogramOptions:
    """Styling for a single histogram series added via ``Histogram.add_histogram``.

    Defaults mirror the previous ``add_histogram`` kwargs: a filled bar with no
    visible edge. For ``histtype="outline"`` the method itself enforces the
    sensible-outline overrides (positive ``edgewidth``, ``facecolor="none"``,
    ``edgecolor="black"``).

    Attributes:
        facecolor (Color): Fill color for histogram bars.
        facealpha (float | None): Optional alpha override for the fill.
        edgecolor (Color): Edge color for histogram bars.
        edgealpha (float | None): Optional alpha override for the edge.
        edgewidth (float): Edge line width in points.
        histtype (HistType): One of "overlay", "stack", "weave", "outline".
        zorder (int): Z-order for layering.
    """

    facecolor: Color = "denim"
    facealpha: float | None = None
    edgecolor: Color = "none"
    edgealpha: float | None = None
    edgewidth: float = 0.0
    histtype: HistType = "overlay"
    zorder: int = 2

    def __post_init__(self) -> None:
        edge_width_value = float(self.edgewidth)
        if not math.isfinite(edge_width_value):
            raise ValueError("edgewidth must be finite.")
        if edge_width_value < 0:
            raise ValueError("edgewidth must be nonnegative.")
        object.__setattr__(self, "edgewidth", edge_width_value)

        resolved_facecolor, resolved_facealpha = resolve_color_and_alpha(
            self.facecolor,
            self.facealpha,
            allow_none=True,
            field="facecolor",
            owner="HistogramOptions",
            logger=logger,
        )
        object.__setattr__(self, "facecolor", resolved_facecolor)
        object.__setattr__(self, "facealpha", resolved_facealpha)

        resolved_edgecolor, resolved_edgealpha = resolve_color_and_alpha(
            self.edgecolor,
            self.edgealpha,
            allow_none=True,
            field="edgecolor",
            owner="HistogramOptions",
            logger=logger,
        )
        object.__setattr__(self, "edgecolor", resolved_edgecolor)
        object.__setattr__(self, "edgealpha", resolved_edgealpha)

        if resolved_edgecolor.lower() == "none" and edge_width_value > 0:
            logger.log(
                level=logging.DEBUG,
                msg=(
                    "HistogramOptions: edgecolor is 'none' but "
                    f"edgewidth is {edge_width_value}>0; setting edgewidth to 0."
                ),
            )
            object.__setattr__(self, "edgewidth", 0.0)

        object.__setattr__(self, "zorder", int(self.zorder))


@dataclass(frozen=True)
class BoxPlotOptions:
    """Styling for a single boxplot dataset added via ``BoxPlot.add_boxplot_dataset``.

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

    facecolor: Color = "denim"
    facealpha: float | None = None
    edgecolor: Color = "black"
    edgealpha: float | None = None
    edgewidth: float = 0.8
    percentiles: tuple[float, float] = (1, 99)
    showfliers: bool = False
    flier_options: PointMarkerOptions = field(default_factory=PointMarkerOptions)
    zorder: int = 1

    def __post_init__(self) -> None:
        percentile_low, percentile_high = self.percentiles
        percentile_low = float(percentile_low)
        percentile_high = float(percentile_high)
        if not (0.0 <= percentile_low <= 100.0 and 0.0 <= percentile_high <= 100.0):
            raise ValueError("percentiles must be within [0, 100].")
        if not (percentile_low < percentile_high):
            raise ValueError("percentiles must satisfy low < high.")

        edge_width_value = float(self.edgewidth)
        if not math.isfinite(edge_width_value):
            raise ValueError("edgewidth must be finite.")
        if edge_width_value < 0:
            raise ValueError("edgewidth must be nonnegative.")
        object.__setattr__(self, "edgewidth", edge_width_value)

        resolved_facecolor, resolved_facealpha = resolve_color_and_alpha(
            self.facecolor,
            self.facealpha,
            allow_none=True,
            field="facecolor",
            owner="BoxPlotOptions",
            logger=logger,
        )
        object.__setattr__(self, "facecolor", resolved_facecolor)
        object.__setattr__(self, "facealpha", resolved_facealpha)

        resolved_edgecolor, resolved_edgealpha = resolve_color_and_alpha(
            self.edgecolor,
            self.edgealpha,
            allow_none=True,
            field="edgecolor",
            owner="BoxPlotOptions",
            logger=logger,
        )
        object.__setattr__(self, "edgecolor", resolved_edgecolor)
        object.__setattr__(self, "edgealpha", resolved_edgealpha)

        if resolved_edgecolor.lower() == "none" and edge_width_value > 0:
            logger.log(
                level=logging.DEBUG,
                msg=(
                    "BoxPlotOptions: edgecolor is 'none' but "
                    f"edgewidth is {edge_width_value}>0; setting edgewidth to 0."
                ),
            )
            object.__setattr__(self, "edgewidth", 0.0)

        object.__setattr__(self, "zorder", int(self.zorder))


@dataclass(frozen=True)
class ViolinPlotOptions:
    """Styling for a single violin dataset added via ``ViolinPlot.add_violinplot_datasets``.

    Attributes:
        facecolor (Color): Fill color for violins.
        facealpha (float | None): Optional alpha override for the fill.
        edgecolor (Color): Edge color for violin outline.
        edgealpha (float | None): Optional alpha override for the edge.
        edgewidth (float): Edge line width.
        zorder (int): Z-order for layering.
    """

    facecolor: Color = "denim"
    facealpha: float | None = None
    edgecolor: Color = "black"
    edgealpha: float | None = None
    edgewidth: float = 0.8
    zorder: int = 1

    def __post_init__(self) -> None:
        edge_width_value = float(self.edgewidth)
        if not math.isfinite(edge_width_value):
            raise ValueError("edgewidth must be finite.")
        if edge_width_value < 0:
            raise ValueError("edgewidth must be nonnegative.")
        object.__setattr__(self, "edgewidth", edge_width_value)

        resolved_facecolor, resolved_facealpha = resolve_color_and_alpha(
            self.facecolor,
            self.facealpha,
            allow_none=True,
            field="facecolor",
            owner="ViolinPlotOptions",
            logger=logger,
        )
        object.__setattr__(self, "facecolor", resolved_facecolor)
        object.__setattr__(self, "facealpha", resolved_facealpha)

        resolved_edgecolor, resolved_edgealpha = resolve_color_and_alpha(
            self.edgecolor,
            self.edgealpha,
            allow_none=True,
            field="edgecolor",
            owner="ViolinPlotOptions",
            logger=logger,
        )
        object.__setattr__(self, "edgecolor", resolved_edgecolor)
        object.__setattr__(self, "edgealpha", resolved_edgealpha)

        if resolved_edgecolor.lower() == "none" and edge_width_value > 0:
            logger.log(
                level=logging.DEBUG,
                msg=(
                    "ViolinPlotOptions: edgecolor is 'none' but "
                    f"edgewidth is {edge_width_value}>0; setting edgewidth to 0."
                ),
            )
            object.__setattr__(self, "edgewidth", 0.0)

        object.__setattr__(self, "zorder", int(self.zorder))


# ---------------------------------------------------------------------------
# SeatsVotes — line and marker subsets.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeatsVotesLineOptions:
    """Styling for the seats-votes curve line in ``SeatsVotes.add_seat_votes_data``.

    Fields ``linewidth`` and ``linealpha`` may be ``None`` to inherit from
    plot-level defaults.

    Attributes:
        linecolor (Color | None): Curve color; ``None`` inherits from caller.
        linealpha (float | None): Optional alpha override.
        linestyle (str): Matplotlib linestyle. Defaults to "-".
        linewidth (float | None): Optional width override; ``None`` inherits.
        zorder (int): Z-order for the curve.
    """

    linecolor: Color | None = None
    linealpha: float | None = None
    linestyle: str = "-"
    linewidth: float | None = None
    zorder: int = 1

    def __post_init__(self) -> None:
        if self.linealpha is not None:
            line_alpha_value = float(self.linealpha)
            if not (0.0 <= line_alpha_value <= 1.0):
                raise ValueError("linealpha must be in [0, 1].")
            object.__setattr__(self, "linealpha", line_alpha_value)

        if self.linewidth is not None:
            line_width_value = float(self.linewidth)
            if not math.isfinite(line_width_value):
                raise ValueError("linewidth must be finite.")
            if line_width_value < 0.0:
                raise ValueError("linewidth must be nonnegative.")
            object.__setattr__(self, "linewidth", line_width_value)

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
    """Styling for the election-result marker in ``SeatsVotes.add_seat_votes_data``.

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
        markerzorder (int): Z-order for the marker.
    """

    markerfacecolor: Color | None = None
    markerfacealpha: float | None = None
    marker: str = "o"
    markersize: float | None = None
    markeredgecolor: Color | None = None
    markeredgealpha: float | None = None
    markeredgewidth: float = 0.0
    markerzorder: int = 2

    def __post_init__(self) -> None:
        if self.markerfacealpha is not None:
            face_alpha_value = float(self.markerfacealpha)
            if not (0.0 <= face_alpha_value <= 1.0):
                raise ValueError("markerfacealpha must be in [0, 1].")
            object.__setattr__(self, "markerfacealpha", face_alpha_value)

        if self.markersize is not None:
            marker_size_value = float(self.markersize)
            if not math.isfinite(marker_size_value):
                raise ValueError("markersize must be finite.")
            if marker_size_value < 0.0:
                raise ValueError("markersize must be nonnegative.")
            object.__setattr__(self, "markersize", marker_size_value)

        if self.markeredgealpha is not None:
            edge_alpha_value = float(self.markeredgealpha)
            if not (0.0 <= edge_alpha_value <= 1.0):
                raise ValueError("markeredgealpha must be in [0, 1].")
            object.__setattr__(self, "markeredgealpha", edge_alpha_value)

        marker_edge_width_value = float(self.markeredgewidth)
        if not math.isfinite(marker_edge_width_value):
            raise ValueError("markeredgewidth must be finite.")
        if marker_edge_width_value < 0.0:
            raise ValueError("markeredgewidth must be nonnegative.")
        object.__setattr__(self, "markeredgewidth", marker_edge_width_value)

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

        object.__setattr__(self, "markerzorder", int(self.markerzorder))


# ---------------------------------------------------------------------------
# SeaLevel — line subset only (markers reuse PointMarkerOptions).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeaLevelLineOptions:
    """Styling for the connecting line in ``SeaLevel.add_sealevel_set``.

    Attributes:
        linecolor (Color): Line color.
        linealpha (float | None): Optional alpha override.
        linewidth (float): Line width in points.
        linestyle (str): Matplotlib linestyle.
        zorder (int): Z-order for the line.
    """

    linecolor: Color = "black"
    linealpha: float | None = None
    linewidth: float = 1.5
    linestyle: str = "-"
    zorder: int = 2

    def __post_init__(self) -> None:
        line_width_value = float(self.linewidth)
        if not math.isfinite(line_width_value):
            raise ValueError("linewidth must be finite.")
        if line_width_value < 0:
            raise ValueError("linewidth must be nonnegative.")
        object.__setattr__(self, "linewidth", line_width_value)

        resolved_linecolor, resolved_linealpha = resolve_color_and_alpha(
            self.linecolor,
            self.linealpha,
            allow_none=True,
            field="linecolor",
            owner="SeaLevelLineOptions",
            logger=logger,
        )
        object.__setattr__(self, "linecolor", resolved_linecolor)
        object.__setattr__(self, "linealpha", resolved_linealpha)

        object.__setattr__(self, "zorder", int(self.zorder))
