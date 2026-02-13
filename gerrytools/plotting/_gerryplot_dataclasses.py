import logging
import math
from dataclasses import dataclass
from typing import Iterable

from gerrytools.colors import resolve_color_and_alpha
from gerrytools.logging import get_logger
from gerrytools.plotting._gerryplot_to_mpl_option_dataclasses import PointMarkerOptions
from gerrytools.typing import Color

logger = get_logger(__name__)


@dataclass(frozen=True)
class PointSetData:
    """A dataclass representing a set of points to be plotted on a boxplot figure.

    Attributes:
        name (str): The name of the point set.
        values_dict (dict[str, float]): A dictionary mapping labels to point values.
        point_data (PointMarkerOptions): The settings for the points.
        x_offset (float | None): An optional absolute x-offset from category center.
    """

    name: str
    values_dict: dict[str, float]  # one value per label
    point_data: PointMarkerOptions
    x_offset: float | None = None  # optional absolute x-offset from category center


@dataclass(frozen=True)
class LineData:
    """Data class representing a line to be drawn on a plot.

    Attributes:
        values (float | Iterable[float]): The position(s) of the line on the axis.
        linecolor (Color): The color of the line.
        linealpha (float | None): The alpha transparency of the line color.
            If None, uses the alpha from the color if specified.
        linestyle (str): The style of the line (e.g., '-', '--', '-.', ':').
        linewidth (float): The width of the line.
        zorder (int): The z-order of the line.
        name (str | None): The name of the line for legend purposes.
    """

    values: float | Iterable[float]
    linecolor: Color = "#cccccc"
    linealpha: float | None = None
    linestyle: str = "-"
    linewidth: float = 1.0
    zorder: int = 3
    name: str | None = None

    def __post_init__(self) -> None:
        lw = float(self.linewidth)
        if lw < 0:
            raise ValueError("LineData.linewidth must be nonnegative.")
        if not math.isfinite(lw):
            raise ValueError("LineData.linewidth must be finite.")
        object.__setattr__(self, "linewidth", lw)

        resolved_lc, resolved_la = resolve_color_and_alpha(
            self.linecolor,
            self.linealpha,
            allow_none=True,
            field="linecolor",
            owner="LineData",
            logger=logger,
        )
        object.__setattr__(self, "linecolor", resolved_lc)
        object.__setattr__(self, "linealpha", resolved_la)

        if resolved_lc.lower() == "none" and lw > 0:
            logger.log(
                level=logging.DEBUG,
                msg=(
                    "LineData: linecolor is 'none' but "
                    f"linewidth is {lw}>0; setting linewidth to 0."
                ),
            )
            object.__setattr__(self, "linewidth", 0.0)

        object.__setattr__(self, "zorder", int(self.zorder))


@dataclass(frozen=True)
class BandData:
    """Data class representing a band to be drawn on a plot.

    Attributes:
        lower_bound (float): The lower bound of the band.
        upper_bound (float): The upper bound of the band.
        bandcolor (Color): The fill color of the band.
        alpha (float | None): The alpha transparency of the band color.
            If None, uses the alpha from the color if specified.
        linecolor (Color | None): The color of the bounding lines of the band.
        linealpha (float | None): The alpha transparency of the bounding lines.
        linestyle (str): The style of the bounding lines (e.g., '-', '--', '-.', ':').
        linewidth (float): The width of the bounding lines.
        zorder (int): The z-order of the band.
        name (str | None): The name of the band for legend purposes.
    """

    lower_bound: float
    upper_bound: float
    bandcolor: Color = "#cccccc"
    bandalpha: float | None = None
    linecolor: Color | None = None
    linealpha: float | None = None
    linestyle: str = "-"
    linewidth: float = 1.0
    zorder: int = 3
    name: str | None = None

    def __post_init__(self) -> None:
        lb, ub = sorted([float(self.lower_bound), float(self.upper_bound)])
        if not (math.isfinite(lb) and math.isfinite(ub)):
            raise ValueError("BandData: lower_bound and upper_bound must both be finite.")
        object.__setattr__(self, "lower_bound", lb)
        object.__setattr__(self, "upper_bound", ub)

        resolved_bc, resolved_ba = resolve_color_and_alpha(
            self.bandcolor,
            self.bandalpha,
            allow_none=True,
            field="bandcolor",
            owner="BandData",
            logger=logger,
        )
        object.__setattr__(self, "bandcolor", resolved_bc)
        object.__setattr__(self, "bandalpha", resolved_ba)

        lw = float(self.linewidth)
        if lw < 0:
            raise ValueError("BandData.linewidth must be nonnegative.")
        if not math.isfinite(lw):
            raise ValueError("BandData.linewidth must be finite.")

        # Default linecolor: follow bandcolor unless band is none (then fallback)
        normalized_line_color = self.linecolor
        if normalized_line_color is None:
            normalized_line_color = resolved_bc
            if isinstance(normalized_line_color, str) and normalized_line_color.lower() == "none":
                normalized_line_color = "#cccccc"

        # Line color + alpha
        resolved_lc, resolved_la = resolve_color_and_alpha(
            normalized_line_color,
            self.linealpha,
            allow_none=True,
            field="linecolor",
            owner="BandData",
            logger=logger,
        )
        object.__setattr__(self, "linecolor", resolved_lc)
        object.__setattr__(self, "linealpha", resolved_la)

        if resolved_lc.lower() == "none" and lw > 0:
            logger.debug(
                "BandData: linecolor is 'none' but linewidth is %s>0; setting linewidth to 0.",
                lw,
            )
            lw = 0.0

        object.__setattr__(self, "linewidth", lw)
        object.__setattr__(self, "zorder", int(self.zorder))
