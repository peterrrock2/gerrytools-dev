from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import matplotlib.colors as mcolors

from gerrytools.colors import resolve_color_and_alpha
from gerrytools.logging import get_logger
from gerrytools.typing import Color

logger = get_logger(__name__)


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
        lw = float(self.markeredgewidth)
        if not math.isfinite(lw):
            raise ValueError("markeredgewidth must be finite")
        if lw < 0:
            raise ValueError("markeredgewidth must be nonnegative")
        object.__setattr__(self, "markeredgewidth", lw)

        size = float(self.markersize)
        if not math.isfinite(size):
            raise ValueError("markersize must be finite")
        if size < 0:
            raise ValueError("markersize must be nonnegative")
        object.__setattr__(self, "markersize", size)

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

        resolved_mec, resolved_mea = resolve_color_and_alpha(
            self.markeredgecolor,
            self.markeredgealpha,
            allow_none=True,
            field="markeredgecolor",
            owner="PointMarkerOptions",
            logger=logger,
        )

        object.__setattr__(self, "markeredgecolor", resolved_mec)
        object.__setattr__(self, "markeredgealpha", resolved_mea)

        if resolved_mec.lower() == "none" and lw > 0:
            logger.log(
                level=logging.DEBUG,
                msg=(
                    "PointMarkerOptions: markeredgecolor is 'none' but "
                    f"markeredgewidth is {lw}>0; setting markeredgewidth to 0."
                ),
            )
            object.__setattr__(self, "markeredgewidth", 0.0)

    def to_mpl_settings_dict(self) -> dict[str, Any]:
        """Convert to Matplotlib kwargs for ``Axes.plot`` marker styling."""
        return {
            "markerfacecolor": mcolors.to_rgba(self.markerfacecolor, alpha=self.markerfacealpha),
            "marker": self.marker,
            "markersize": self.markersize,
            "markeredgecolor": mcolors.to_rgba(self.markeredgecolor, alpha=self.markeredgealpha),
            "markeredgewidth": self.markeredgewidth,
            "zorder": self.zorder,
        }

    def to_mpl_scatter_settings_dict(self) -> dict[str, Any]:
        """Convert to Matplotlib kwargs for ``Axes.scatter`` marker styling."""
        return {
            "marker": self.marker,
            "s": self.markersize**2,
            "edgecolor": mcolors.to_rgba(self.markeredgecolor, alpha=self.markeredgealpha),
            "linewidths": self.markeredgewidth,
            "zorder": self.zorder,
        }
