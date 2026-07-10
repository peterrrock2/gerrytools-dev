from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

from gerrytools.colors import resolve_color_and_alpha
from gerrytools.logging import get_logger
from gerrytools.plotting.utils import _resolve_color_clamped_width, _validated_nonneg_finite
from gerrytools.typing import Color, MplRGBAColor

logger = get_logger(__name__)


class PlotMarkerKwargs(TypedDict):
    """Marker kwargs emitted by ``PointMarkerOptions.to_mpl_settings_dict``."""

    markerfacecolor: MplRGBAColor
    marker: str
    markersize: float
    markeredgecolor: MplRGBAColor
    markeredgewidth: float
    zorder: int


class ScatterMarkerKwargs(TypedDict):
    """Marker kwargs emitted by ``PointMarkerOptions.to_mpl_scatter_settings_dict``."""

    marker: str
    s: float
    edgecolor: MplRGBAColor
    linewidths: float
    zorder: int


@dataclass(slots=True)
class PointMarkerOptions:
    """Settings for points on a matplotlib plot (or for functions that use similar artists).

    Attributes:
        markerfacecolor (Color): The fill color of the marker. Defaults to "none".
        markerfacealpha (float | None): The alpha transparency of the marker face color.
            If None, uses the alpha from the color if specified. Defaults to None.
        marker (str): The marker style. Defaults to "o".
        markersize (float): The size of the marker. Defaults to 6.0.
        markeredgecolor (Color): The edge color of the marker. Defaults to "black".
        markeredgealpha (float | None): The alpha transparency of the marker edge color.
            If None, uses the alpha from the color if specified. Defaults to None.
        markeredgewidth (float): The width of the marker edge. Defaults to 0.6.
        zorder (int): The z-order of the marker. Defaults to 4.
    """

    markerfacecolor: Color = "none"
    markerfacealpha: float | None = None
    marker: str = "o"
    markersize: float = 6.0
    markeredgecolor: Color = "black"
    markeredgealpha: float | None = None
    markeredgewidth: float = 0.6
    zorder: int = 4

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "markeredgewidth",
            _validated_nonneg_finite(self.markeredgewidth, field="markeredgewidth"),
        )
        object.__setattr__(
            self, "markersize", _validated_nonneg_finite(self.markersize, field="markersize")
        )

        resolved_mfc, resolved_mfa = resolve_color_and_alpha(
            self.markerfacecolor,
            self.markerfacealpha,
            allow_none=True,
            field="markerfacecolor",
            owner="PointMarkerOptions",
            logger=logger,
        )

        object.__setattr__(self, "markerfacecolor", resolved_mfc)
        object.__setattr__(self, "markerfacealpha", resolved_mfa)

        resolved_mec, resolved_mea, clamped_edge_width = _resolve_color_clamped_width(
            self.markeredgecolor,
            self.markeredgealpha,
            self.markeredgewidth,
            color_field="markeredgecolor",
            width_field="markeredgewidth",
            owner="PointMarkerOptions",
            log=logger,
        )

        object.__setattr__(self, "markeredgecolor", resolved_mec)
        object.__setattr__(self, "markeredgealpha", resolved_mea)
        object.__setattr__(self, "markeredgewidth", clamped_edge_width)

    def to_mpl_settings_dict(self) -> PlotMarkerKwargs:
        """Convert to Matplotlib kwargs for ``Axes.plot`` marker styling."""
        return {
            "markerfacecolor": mcolors.to_rgba(self.markerfacecolor, alpha=self.markerfacealpha),
            "marker": self.marker,
            "markersize": self.markersize,
            "markeredgecolor": mcolors.to_rgba(self.markeredgecolor, alpha=self.markeredgealpha),
            "markeredgewidth": self.markeredgewidth,
            "zorder": self.zorder,
        }

    def to_mpl_scatter_settings_dict(self) -> ScatterMarkerKwargs:
        """Convert to Matplotlib kwargs for ``Axes.scatter`` marker styling."""
        return {
            "marker": self.marker,
            "s": self.markersize**2,
            "edgecolor": mcolors.to_rgba(self.markeredgecolor, alpha=self.markeredgealpha),
            "linewidths": self.markeredgewidth,
            "zorder": self.zorder,
        }


def _marker_legend_handle(marker_options: PointMarkerOptions, label: str | None) -> Line2D:
    """Build the standard line-less marker legend handle for a point set."""
    return Line2D(
        [0],
        [0],
        linestyle="none",
        label=label,
        **marker_options.to_mpl_settings_dict(),
    )
