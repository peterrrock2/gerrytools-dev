import logging
import math
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Real
from typing import Any, Iterable, Literal

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from gerrytools.colors import resolve_color_and_alpha
from gerrytools.logging import get_logger
from gerrytools.plotting.utils import _coerce_real_iter
from gerrytools.typing import Color, TickType

logger = get_logger(__name__)


@dataclass(frozen=True)
class PointMarkerSettings:
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

        s = float(self.markersize)
        if not math.isfinite(s):
            raise ValueError("markersize must be finite")
        if s < 0:
            raise ValueError("markersize must be nonnegative")
        object.__setattr__(self, "markersize", s)

        resolved_mfc, resolved_mfa = resolve_color_and_alpha(
            self.markerfacecolor,
            self.markerfacealpha,
            allow_none=True,
            field="markerfacecolor",
            owner="PointMarkerSettings",
            logger=logger,
        )

        object.__setattr__(self, "markerfacecolor", resolved_mfc)
        object.__setattr__(self, "markerfacealpha", resolved_mfa)

        resolved_mec, resolved_mea = resolve_color_and_alpha(
            self.markeredgecolor,
            self.markeredgealpha,
            allow_none=True,
            field="markeredgecolor",
            owner="PointMarkerSettings",
            logger=logger,
        )

        object.__setattr__(self, "markeredgecolor", resolved_mec)
        object.__setattr__(self, "markeredgealpha", resolved_mea)

        if resolved_mec.lower() == "none" and lw > 0:
            logger.log(
                level=logging.DEBUG,
                msg=(
                    "PointMarkerSettings: markeredgecolor is 'none' but "
                    f"markeredgewidth is {lw}>0; setting markeredgewidth to 0."
                ),
            )
            object.__setattr__(self, "markeredgewidth", 0.0)

    def to_mpl_settings_dict(self) -> dict[str, Any]:
        """Convert the PointMarkerSettings to a dictionary.

        Returns:
            dict[str, Any]: A dictionary representation of the PointMarkerSettings that
                can be passed to Matplotlib plot functions.
        """
        # Matplotlib alpha applies to the entire marker, so we need to
        # apply alpha to the facecolor and edgecolor separately to get things to
        # work as expected.
        return {
            "markerfacecolor": mcolors.to_rgba(self.markerfacecolor, alpha=self.markerfacealpha),
            "marker": self.marker,
            "markersize": self.markersize,
            "markeredgecolor": mcolors.to_rgba(self.markeredgecolor, alpha=self.markeredgealpha),
            "markeredgewidth": self.markeredgewidth,
            "zorder": self.zorder,
        }


@dataclass(frozen=True)
class PointSetData:
    """A dataclass representing a set of points to be plotted on a boxplot figure.

    Attributes:
        name (str): The name of the point set.
        values_dict (dict[str, float]): A dictionary mapping labels to point values.
        point_data (PointMarkerSettings): The settings for the points.
        x_offset (float | None): An optional absolute x-offset from category center.
    """

    name: str
    values_dict: dict[str, float]  # one value per label
    point_data: PointMarkerSettings
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


@dataclass(frozen=True)
class TickStyle:
    """Data class representing the style of axis ticks.

    Attributes:
        size (float | int): The size of the ticks. Defaults to 10.
        rotation (float | int): The rotation angle of the tick labels in degrees. Defaults to 0.
        fontcolor (Color): The color of the tick labels. Defaults to "black".
        fontalpha (float | None): The alpha transparency of the tick label color.
            If None, uses the alpha from the color if specified. Defaults to None.
        tickcolor (Color): The color of the ticks. Defaults to "black".
        tickalpha (float | None): The alpha transparency of the tick color.
            If None, uses the alpha from the color if specified. Defaults to None.
        fontweight (str): The weight of the tick label font. Defaults to "normal".
        fontstyle (str): The style of the tick label font. Defaults to "normal".
        fontfamily (str): The family of the tick label font. Defaults to "sans-serif".
        ticktype (TickType): The type of ticks to apply the style to. Defaults to "major".
    """

    size: float | int = 10
    rotation: float | int = 0
    fontcolor: Color = "black"
    fontalpha: float | None = None
    tickcolor: Color = "black"
    tickalpha: float | None = None
    fontweight: str = "normal"
    fontstyle: str = "normal"
    fontfamily: str = "sans-serif"
    ticktype: TickType = "major"

    def __post_init__(self) -> None:
        if not isinstance(self.size, (int, float)):
            raise TypeError("TickStyle.size must be a float or int.")
        if not math.isfinite(self.size):
            raise ValueError("TickStyle.size must be finite.")
        if not float(self.size) >= 0:
            raise ValueError("TickStyle.size must be nonnegative.")

        resolved_fc, resolved_fa = resolve_color_and_alpha(
            self.fontcolor,
            self.fontalpha,
            allow_none=True,
            field="fontcolor",
            owner="TickStyle",
            logger=logger,
        )
        object.__setattr__(self, "fontcolor", resolved_fc)
        object.__setattr__(self, "fontalpha", resolved_fa)

        resolved_tc, resolved_ta = resolve_color_and_alpha(
            self.tickcolor,
            self.tickalpha,
            allow_none=True,
            field="tickcolor",
            owner="TickStyle",
            logger=logger,
        )
        object.__setattr__(self, "tickcolor", resolved_tc)
        object.__setattr__(self, "tickalpha", resolved_ta)

        if self.ticktype not in ("major", "minor", "both"):
            raise ValueError("TickStyle.ticktype must be 'major', 'minor', or 'both'.")


@dataclass
class LegendOptions:
    """A dataclass representing options for the legend in a boxplot figure.

    This is a restricted subset of the options available in Matplotlib's legend function.

    Attributes:
        legend_loc (str | int): The location of the legend. Defaults to "best".
        legend_bbox_to_anchor (tuple[float, float] | None): The bounding box anchor
            for the legend. Defaults to None.
        ncols (int): The number of columns in the legend. Defaults to 1.
        fontsize (float | str | None): The font size of the legend text. Defaults to None.

    """

    loc: str | int = "best"
    bbox_to_anchor: tuple[float, float] | tuple[float, float, float, float] | None = None
    ncols: int = 1
    fontsize: float | str | None = None
    frameon: bool = True
    fancybox: bool = False
    shadow: bool = False
    framealpha: float | None = None
    facecolor: Color | None = None
    edgecolor: Color | None = None
    title: str | None = None
    alignment: Literal["center", "left", "right"] = "center"
    labelspacing: float = 0.5
    columnspacing: float = 2.0

    def to_dict(self) -> dict[str, Any]:
        """Convert the LegendOptions to a dictionary.

        Returns:
            dict[str, Any]: A dictionary representation of the LegendOptions that
                can be passed to Matplotlib's legend function. Returns only fields
                that are not None.
        """
        output = {}
        for field_name, field_value in self.__dict__.items():
            if field_value is not None:
                output[field_name] = field_value

        return output


@dataclass(frozen=True)
class AxisLabelStyle:
    """Dataclass meant to mirror some matplotlib styling options for axis labels.

    Attributes:
        fontsize (float | int | None): Font size for the axis label text.
        fontweight (str | None): Font weight (e.g., "normal", "bold").
        fontstyle (str | None): Font style (e.g., "normal", "italic").
        fontfamily (str | None): Font family (e.g., "sans-serif", "serif").
        fontcolor (Color): Axis label text color.
        fontalpha (float | None): Axis label text alpha. If None, uses alpha from color if
            specified.
        labelpad (float | None): Padding between the label and the axis (in points).
    """

    fontsize: float | int | None = None
    fontweight: str | None = None
    fontstyle: str | None = None
    fontfamily: str | None = None

    fontcolor: Color = "black"
    fontalpha: float | None = None

    labelpad: float | None = None

    def __post_init__(self) -> None:
        if self.fontsize is not None:
            if not isinstance(self.fontsize, (int, float)):
                raise TypeError("AxisLabelStyle.fontsize must be a float or int.")
            size = float(self.fontsize)
            if not math.isfinite(size):
                raise ValueError("AxisLabelStyle.fontsize must be finite.")
            if size < 0:
                raise ValueError("AxisLabelStyle.fontsize must be nonnegative.")
            object.__setattr__(self, "fontsize", self.fontsize)

        if self.labelpad is not None:
            if not isinstance(self.labelpad, (int, float)):
                raise TypeError("AxisLabelStyle.labelpad must be a float or int.")
            pad = float(self.labelpad)
            if not math.isfinite(pad):
                raise ValueError("AxisLabelStyle.labelpad must be finite.")
            if pad < 0:
                raise ValueError("AxisLabelStyle.labelpad must be nonnegative.")
            object.__setattr__(self, "labelpad", pad)

        resolved_c, resolved_a = resolve_color_and_alpha(
            self.fontcolor,
            self.fontalpha,
            allow_none=True,
            field="fontcolor",
            owner="AxisLabelStyle",
            logger=logger,
        )
        object.__setattr__(self, "fontcolor", resolved_c)
        object.__setattr__(self, "fontalpha", resolved_a)

    def to_mpl_settings_dict(self) -> dict[str, Any]:
        """Convert to Matplotlib keyword dictionary for ``Axes.set_xlabel`` /
        ``Axes.set_ylabel``.
        """
        settings_dict: dict[str, Any] = {
            "color": mcolors.to_rgba(self.fontcolor, alpha=self.fontalpha),
        }
        if self.fontsize is not None:
            settings_dict["fontsize"] = self.fontsize
        if self.fontweight is not None:
            settings_dict["fontweight"] = self.fontweight
        if self.fontstyle is not None:
            settings_dict["fontstyle"] = self.fontstyle
        if self.fontfamily is not None:
            settings_dict["fontfamily"] = self.fontfamily
        if self.labelpad is not None:
            settings_dict["labelpad"] = self.labelpad
        return settings_dict


@dataclass(frozen=True)
class TitleStyle:
    """Dataclass meant to mirror some matplotlib styling options for an axes title.

    Attributes:
        fontsize (float | int | None): Font size for the title text.
        fontweight (str | None): Font weight (e.g., "normal", "bold").
        fontstyle (str | None): Font style (e.g., "normal", "italic").
        fontfamily (str | None): Font family (e.g., "sans-serif", "serif").
        fontcolor (Color): Title text color.
        fontalpha (float | None): Title text alpha. If None, uses alpha from color if specified.
        loc (Literal["left", "center", "right"] | None): Title location.
        pad (float | None): Padding between the title and the axes (in points).
    """

    fontsize: float | int | None = None
    fontweight: str | None = None
    fontstyle: str | None = None
    fontfamily: str | None = None

    fontcolor: Color = "black"
    fontalpha: float | None = None

    loc: Literal["left", "center", "right"] | None = None
    pad: float | None = None

    def __post_init__(self) -> None:
        if self.fontsize is not None:
            if not isinstance(self.fontsize, (int, float)):
                raise TypeError("TitleStyle.fontsize must be a float or int.")
            size = float(self.fontsize)
            if not math.isfinite(size):
                raise ValueError("TitleStyle.fontsize must be finite.")
            if size < 0:
                raise ValueError("TitleStyle.fontsize must be nonnegative.")
            object.__setattr__(self, "fontsize", self.fontsize)

        if self.pad is not None:
            if not isinstance(self.pad, (int, float)):
                raise TypeError("TitleStyle.pad must be a float or int.")
            pad = float(self.pad)
            if not math.isfinite(pad):
                raise ValueError("TitleStyle.pad must be finite.")
            if pad < 0:
                raise ValueError("TitleStyle.pad must be nonnegative.")
            object.__setattr__(self, "pad", pad)

        if self.loc is not None and self.loc not in ("left", "center", "right"):
            raise ValueError("TitleStyle.loc must be one of {'left','center','right'}.")

        resolved_c, resolved_a = resolve_color_and_alpha(
            self.fontcolor,
            self.fontalpha,
            allow_none=True,
            field="fontcolor",
            owner="TitleStyle",
            logger=logger,
        )
        object.__setattr__(self, "fontcolor", resolved_c)
        object.__setattr__(self, "fontalpha", resolved_a)

    def to_mpl_settings_dict(self) -> dict[str, Any]:
        """Convert to Matplotlib kwargs for ``Axes.set_title``."""
        settings_dict: dict[str, Any] = {
            "color": mcolors.to_rgba(self.fontcolor, alpha=self.fontalpha),
        }
        if self.fontsize is not None:
            settings_dict["fontsize"] = self.fontsize
        if self.fontweight is not None:
            settings_dict["fontweight"] = self.fontweight
        if self.fontstyle is not None:
            settings_dict["fontstyle"] = self.fontstyle
        if self.fontfamily is not None:
            settings_dict["fontfamily"] = self.fontfamily
        if self.loc is not None:
            settings_dict["loc"] = self.loc
        if self.pad is not None:
            settings_dict["pad"] = self.pad
        return settings_dict


class GerryPlotBase(ABC):
    """Abstract base class for GerryPlot plotting classes."""

    def __init__(
        self,
        figure_size: tuple[float, float] = (10, 6),
        dpi: int = 300,
        *,
        include_legend: bool = True,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
    ) -> None:
        """Initialize a GerryPlotBase instance.

        Args:
            figure_size (tuple[float, float], optional): The size of the figure in inches.
                Defaults to (10, 6).
            dpi (int, optional): The dots per inch (DPI) of the figure. Defaults to 300.

        Kwargs:
            include_legend (bool, optional): Whether to include a legend in the plot.
                Defaults to True.
            xlabel (str | None, optional): The label for the x-axis. Defaults to None.
            ylabel (str | None, optional): The label for the y-axis. Defaults to None.
            title (str | None, optional): The title of the plot. Defaults to None.
        """
        self.fig, self._ax = plt.subplots(figsize=figure_size, dpi=dpi)

        self.include_legend = include_legend
        self._legend_options = LegendOptions(loc="center left", bbox_to_anchor=(1.01, 0.5))

        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title

        self._xlabel_style: AxisLabelStyle | None = None
        self._ylabel_style: AxisLabelStyle | None = None
        self._title_style: TitleStyle | None = None

        self._x_tick_locations: list[float] | None = None
        self._x_tick_labels: list[str] | None = None
        self._x_limits: tuple[float, float] | None = None
        self._x_tick_style: TickStyle | None = None

        self._y_tick_locations: list[float] | None = None
        self._y_tick_labels: list[str] | None = None
        self._y_limits: tuple[float, float] | None = None
        self._y_tick_style: TickStyle | None = None

        self._vertical_lines: list[LineData] = []
        self._vertical_bands: list[BandData] = []
        self._horizontal_lines: list[LineData] = []
        self._horizontal_bands: list[BandData] = []

        self._frame_visibility: dict[str, bool] = {
            "top": True,
            "right": True,
            "bottom": True,
            "left": True,
        }

        self._finalizer = weakref.finalize(self, plt.close, self.fig)

    def add_vertical_lines(
        self,
        x_values: float | Iterable[float],
        *,
        linecolor: Color = "#cccccc",
        linealpha: float | None = None,
        linestyle: str = "-",
        linewidth: float = 1.0,
        zorder: int = 3,
        name: str | None = None,
    ) -> None:
        """Add a vertical line to the figure.

        Args:
            x_values (float | Iterable[float]): The x-value(s) where the vertical line(s) should be
                drawn.

        Kwargs:
            linecolor (Color, optional): The color of the vertical line. Defaults to "#cccccc".
            linealpha (float | None, optional): The alpha transparency of the vertical line.
                Defaults to None in which case the alpha from linecolor is used if specified.
            linestyle (str, optional): The linestyle of the vertical line. Defaults to "-".
            linewidth (float, optional): The width of the vertical line. Defaults to 1.0.
            zorder (int, optional): The z-order of the vertical line. Defaults to 3.
            name (str | None, optional): The name of the line for legend purposes. Defaults to None.

        Returns:
            None
        """
        if isinstance(x_values, (str, bytes)):
            raise TypeError("x_values must be a number or an iterable of numbers, not a string.")
        if isinstance(x_values, bool):
            raise TypeError("x_values must be a number or an iterable of numbers, not a bool.")
        # Safe to shadow here because we pass ints and floats by value not object reference
        if isinstance(x_values, Real):
            x_values = [float(x_values)]

        xs = _coerce_real_iter(x_values, field="x_values")
        self._vertical_lines.append(
            LineData(
                values=xs,
                linecolor=linecolor,
                linealpha=linealpha,
                linestyle=linestyle,
                linewidth=float(linewidth),
                zorder=zorder,
                name=name,
            )
        )

    def add_vertical_band(
        self,
        x_low: float,
        x_high: float,
        *,
        bandcolor: Color = "#cccccc",
        bandalpha: float | None = None,
        linecolor: Color | None = None,
        linealpha: float | None = None,
        linestyle: str = "-",
        linewidth: float = 1.0,
        zorder: int = 3,
        name: str | None = None,
    ) -> None:
        """Add a vertical band to the figure.

        Args:
            x_low (float): The lower x-value of the vertical band.
            x_high (float): The upper x-value of the vertical band.

        Kwargs:
            bandcolor (Color, optional): The fill color of the band. Defaults to "#cccccc".
            bandalpha (float | None, optional): The alpha transparency of the band. Defaults to None.
            linecolor (Color | None, optional): The color of the bounding lines of the band.
                If set to None and bandcolor is also None, defaults to "#cccccc".
                If set to None and bandcolor is not None, defaults to bandcolor.
                Defaults to None.
            linealpha (float | None, optional): The alpha transparency of the bounding lines.
                Defaults to None which uses the alpha from linecolor if specified.
            linestyle (str, optional): The linestyle of the bounding lines of the band.
                Defaults to "-".
            linewidth (float, optional): The width of the bounding lines of the band.
                Defaults to 1.0.
            zorder (int, optional): The z-order of the band. Defaults to 3.
            name (str | None, optional): The name of the band for legend purposes. Defaults to None.

        Returns:
            None
        """
        self._vertical_bands.append(
            BandData(
                lower_bound=min(x_low, x_high),
                upper_bound=max(x_low, x_high),
                bandcolor=bandcolor,
                bandalpha=bandalpha,
                linecolor=linecolor,
                linealpha=linealpha,
                linestyle=linestyle,
                linewidth=linewidth,
                zorder=zorder,
                name=name,
            )
        )

    def add_horizontal_lines(
        self,
        y_values: float | Iterable[float],
        *,
        linecolor: Color = "#cccccc",
        linealpha: float | None = None,
        linestyle: str = "-",
        linewidth: float = 1.0,
        zorder: int = 4,
        name: str | None = None,
    ) -> None:
        """Add a horizontal line to the figure.

        Args:
            y_values (float | Iterable[float]): The y-value(s) where the horizontal line(s) should
                be drawn.

        Kwargs:
            linecolor (Color, optional): The color of the horizontal line. Defaults to "#cccccc".
            linealpha (float | None, optional): The alpha transparency of the horizontal line.
                Defaults to None in which case the alpha from linecolor is used if specified.
            linestyle (str, optional): The linestyle of the horizontal line. Defaults to "-".
            linewidth (float, optional): The width of the horizontal line. Defaults to 1.0.
            zorder (int, optional): The z-order of the horizontal line. Defaults to 4.
            name (str | None, optional): The name of the line for legend purposes. Defaults to None.

        Returns:
            None
        """
        if isinstance(y_values, (str, bytes)):
            raise TypeError("y_values must be a number or an iterable of numbers, not a string.")
        if isinstance(y_values, bool):
            raise TypeError("y_values must be a number or an iterable of numbers, not a bool.")
        # Safe to shadow here because we pass ints and floats by value not object reference
        if isinstance(y_values, Real):
            y_values = [float(y_values)]

        ys = _coerce_real_iter(y_values, field="y_values")

        self._horizontal_lines.append(
            LineData(
                values=ys,
                linecolor=linecolor,
                linealpha=linealpha,
                linestyle=linestyle,
                linewidth=float(linewidth),
                zorder=zorder,
                name=name,
            )
        )
        return

    def add_horizontal_band(
        self,
        y_low: float,
        y_high: float,
        *,
        bandcolor: Color = "#cccccc",
        bandalpha: float | None = None,
        linecolor: Color | None = None,
        linealpha: float | None = None,
        linestyle: str = "-",
        linewidth: float = 1.0,
        zorder: int = 4,
        name: str | None = None,
    ) -> None:
        """Add a horizontal band to the figure.

        Args:
            y_low (float): The lower y-value of the horizontal band.
            y_high (float): The upper y-value of the horizontal band.

        Kwargs:
            bandcolor (Color | None, optional): The fill color of the band. Defaults to "#cccccc".
            bandalpha (float | None, optional): The alpha transparency of the band. Defaults to None
            linecolor (Color | None, optional): The color of the bounding lines of the band.
                If set to None and bandcolor is also None, defaults to "#cccccc".
                If set to None and bandcolor is not None, defaults to bandcolor.
                Defaults to None.
            linealpha (float | None, optional): The alpha transparency of the bounding lines.
                Defaults to None which uses the alpha from linecolor if specified.
            linestyle (str, optional): The linestyle of the bounding lines of the band.
                Defaults to "-".
            linewidth (float, optional): The width of the bounding lines of the band.
                Defaults to 1.0.
            zorder (int, optional): The z-order of the band. Defaults to 4.
            name (str | None, optional): The name of the band for legend purposes. Defaults to None.

        Returns:
            None
        """
        self._horizontal_bands.append(
            BandData(
                lower_bound=min(y_low, y_high),
                upper_bound=max(y_low, y_high),
                bandcolor=bandcolor,
                bandalpha=bandalpha,
                linecolor=linecolor,
                linealpha=linealpha,
                linestyle=linestyle,
                linewidth=linewidth,
                zorder=zorder,
                name=name,
            )
        )

    def clear_vertical_lines_and_bands(self) -> None:
        """Clear all vertical lines and bands from the figure."""
        self._vertical_lines.clear()
        self._vertical_bands.clear()

    def clear_horizontal_lines_and_bands(self) -> None:
        """Clear all horizontal lines and bands from the figure."""
        self._horizontal_lines.clear()
        self._horizontal_bands.clear()

    def _default_x_tick_locations(self) -> list[float] | None:
        return None

    def _default_x_tick_labels(self, tick_locations: list[float]) -> list[str] | None:
        """Get default x-tick labels for the categorical data."""
        return None

    def _set_x_axis(self) -> None:
        """Set x-axis limits, ticks, and labels in the plot."""
        x_limits = self._x_limits if self._x_limits is not None else self._ax.get_xlim()
        self._ax.set_xlim(x_limits)

        if self._x_tick_locations is not None:
            x_tick_locations = list(self._x_tick_locations)
        else:
            default_locs = self._default_x_tick_locations()
            x_tick_locations = (
                list(default_locs) if default_locs is not None else self._ax.get_xticks().tolist()
            )

        self._ax.set_xticks(x_tick_locations)

        if self._x_tick_labels == []:
            self._ax.tick_params(axis="x", labelbottom=False)
            return

        # If user didn't provide labels, allow subclass defaults
        if self._x_tick_labels is None:
            labels = self._default_x_tick_labels(x_tick_locations)
            if labels is None:
                return
            if len(labels) != len(x_tick_locations):
                raise ValueError(
                    f"Expected {len(x_tick_locations)} x tick labels, got {len(labels)}."
                )
            self._ax.set_xticklabels(list(labels))
            return

        # User provided labels
        if len(self._x_tick_labels) != len(x_tick_locations):
            raise ValueError(
                f"Expected {len(x_tick_locations)} x tick labels, got {len(self._x_tick_labels)}."
            )
        self._ax.set_xticklabels(list(self._x_tick_labels))

    def _set_y_axis(self) -> None:
        """Set y-axis limits, ticks, and labels in the plot."""
        y_limits = self._y_limits if self._y_limits is not None else self._ax.get_ylim()
        self._ax.set_ylim(y_limits)

        if self._y_tick_locations is not None:
            y_tick_locations = list(self._y_tick_locations)
            self._ax.set_yticks(y_tick_locations)
        else:
            y_tick_locations = self._ax.get_yticks().tolist()

        if self._y_tick_labels is None:
            return

        if self._y_tick_labels == []:
            self._ax.tick_params(axis="y", labelleft=False)
            return

        if self._y_tick_locations is None:
            self._ax.set_yticks(y_tick_locations)

        if len(self._y_tick_labels) != len(y_tick_locations):
            raise ValueError(
                f"Expected {len(y_tick_locations)} y tick labels, got {len(self._y_tick_labels)}."
            )

        self._ax.set_yticklabels(list(self._y_tick_labels))

    def update_xtick_values(
        self, *, locations: list[float] | None = None, labels: list[str] | None = None
    ) -> None:
        """Update x-tick locations and/or labels.

        Overrides existing values if provided.

        Kwargs:
            locations (list[float] | None, optional): New x-tick locations. Defaults
                to None.
            labels (list[str] | None, optional): New x-tick labels. Defaults to
                None.

        Raises:
            ValueError: If the lengths of provided locations and labels do not match
                existing values.

        Returns:
            None
        """
        if locations is None and labels is None:
            return

        if locations is not None and labels is not None:
            if (locations == [] and labels not in (None, [])) or (
                labels == [] and locations not in (None, [])
            ):
                raise ValueError(
                    "If clearing ticks/labels, clear both (locations=[] and labels=[])."
                )

            if labels != [] and locations != [] and len(locations) != len(labels):
                raise ValueError(
                    f"Locations length {len(locations)} does not match labels length {len(labels)}."
                )
            self._x_tick_locations = list(locations)
            self._x_tick_labels = list(labels)
            return

        if locations is not None:
            if (
                self._x_tick_labels is not None
                and self._x_tick_labels != []
                and locations != []
                and len(locations) != len(self._x_tick_labels)
            ):
                raise ValueError(
                    f"Locations length {len(locations)} does not match existing labels length "
                    f"{len(self._x_tick_labels)}."
                )
            if locations == [] and labels is None:
                self._x_tick_locations = []
                self._x_tick_labels = []
                return
            self._x_tick_locations = list(locations)
            return

        if labels is not None:
            if labels == []:
                self._x_tick_labels = []
                return

            if (
                self._x_tick_locations is not None
                and self._x_tick_locations != []
                and len(labels) != len(self._x_tick_locations)
            ):
                raise ValueError(
                    f"Labels length {len(labels)} does not match existing locations length "
                    f"{len(self._x_tick_locations)}."
                )
            self._x_tick_labels = list(labels)
            return

    def update_ytick_values(
        self, *, locations: list[float] | None = None, labels: list[str] | None = None
    ) -> None:
        """Update y-tick locations and/or labels.

        Overrides existing values if provided.

        Kwargs:
            locations (list[float] | None, optional): New y-tick locations. Defaults
                to None.
            labels (list[str] | None, optional): New y-tick labels. Defaults to
                None.

        Raises:
            ValueError: If the lengths of provided locations and labels do not match
                existing values.

        Returns:
            None
        """
        if locations is None and labels is None:
            return

        if locations is not None and labels is not None:
            if (locations == [] and labels not in (None, [])) or (
                labels == [] and locations not in (None, [])
            ):
                raise ValueError(
                    "If clearing ticks/labels, clear both (locations=[] and labels=[])."
                )

            if labels != [] and locations != [] and len(locations) != len(labels):
                raise ValueError(
                    f"Locations length {len(locations)} does not match labels length {len(labels)}."
                )
            self._y_tick_locations = list(locations)
            self._y_tick_labels = list(labels)
            return

        if locations is not None:
            if (
                self._y_tick_labels is not None
                and self._y_tick_labels != []
                and locations != []
                and len(locations) != len(self._y_tick_labels)
            ):
                raise ValueError(
                    f"Locations length {len(locations)} does not match existing labels length "
                    f"{len(self._y_tick_labels)}."
                )
            if locations == [] and labels is None:
                self._y_tick_locations = []
                self._y_tick_labels = []
                return
            self._y_tick_locations = list(locations)
            return

        if labels is not None:
            if labels == []:
                self._y_tick_labels = []
                return
            if (
                self._y_tick_locations is not None
                and self._y_tick_locations != []
                and len(labels) != len(self._y_tick_locations)
            ):
                raise ValueError(
                    f"Labels length {len(labels)} does not match existing locations length "
                    f"{len(self._y_tick_locations)}."
                )
            self._y_tick_labels = list(labels)
            return

    @staticmethod
    def _apply_ticklabel_textprops(
        labels,
        *,
        fontweight: str | None = None,
        fontstyle: str | None = None,
        fontfamily: str | None = None,
    ) -> None:
        """Apply text properties to tick labels.

        Args:
            labels: List of tick label objects.

        Kwargs:
            fontweight (str | None, optional): Font weight to apply. Defaults to None.
            fontstyle (str | None, optional): Font style to apply. Defaults to None.
            fontfamily (str | None, optional): Font family to apply. Defaults to None.

        Returns:
            None
        """
        # These are matplotlib.text.Text objects.
        for text in labels:
            if fontweight is not None:
                text.set_fontweight(fontweight)
            if fontstyle is not None:
                text.set_fontstyle(fontstyle)
            if fontfamily is not None:
                text.set_fontfamily(fontfamily)

    def _apply_tick_style(self, axis: Literal["x", "y", "both"], style: TickStyle) -> None:
        """Apply tick style to the specified axis.

        Args:
            axis (Literal["x", "y", "both"]): The axis to apply the style to.
            style (TickStyle): The tick style to apply.

        Returns:
            None
        """
        # Tick marks + tick label basics
        label_color_resolved = mcolors.to_rgba(style.fontcolor, alpha=style.fontalpha)
        tick_color_resolved = mcolors.to_rgba(style.tickcolor, alpha=style.tickalpha)
        self._ax.tick_params(
            axis=axis,
            which=style.ticktype,
            labelsize=style.size,
            rotation=style.rotation,
            labelcolor=label_color_resolved,
            color=tick_color_resolved,
        )

        # Tick label text styling (weight/style/family)
        if axis in ("x", "both"):
            if style.ticktype in ("major", "both"):
                self._apply_ticklabel_textprops(
                    self._ax.get_xticklabels(minor=False),
                    fontweight=style.fontweight,
                    fontstyle=style.fontstyle,
                    fontfamily=style.fontfamily,
                )
            if style.ticktype in ("minor", "both"):
                self._apply_ticklabel_textprops(
                    self._ax.get_xticklabels(minor=True),
                    fontweight=style.fontweight,
                    fontstyle=style.fontstyle,
                    fontfamily=style.fontfamily,
                )

        if axis in ("y", "both"):
            if style.ticktype in ("major", "both"):
                self._apply_ticklabel_textprops(
                    self._ax.get_yticklabels(minor=False),
                    fontweight=style.fontweight,
                    fontstyle=style.fontstyle,
                    fontfamily=style.fontfamily,
                )
            if style.ticktype in ("minor", "both"):
                self._apply_ticklabel_textprops(
                    self._ax.get_yticklabels(minor=True),
                    fontweight=style.fontweight,
                    fontstyle=style.fontstyle,
                    fontfamily=style.fontfamily,
                )

    def _apply_deferred_tick_styles(self) -> None:
        """Apply the tick styles if they were set.

        To be called after the axes have been fully configured so that any call to ax.clear()
        does not wipe out the tick styles.

        Returns:
            None
        """
        if self._x_tick_style is not None:
            self._apply_tick_style("x", self._x_tick_style)
        if self._y_tick_style is not None:
            self._apply_tick_style("y", self._y_tick_style)

    def _apply_deferred_label_styles(self) -> None:
        """Apply xlabel/ylabel/title text and any configured styles.

        Designed to run after ``_build_plot()`` so that any subclass call to ``ax.clear()``
        doesn't wipe out label/title configuration.
        """
        if self.xlabel is not None:
            kwargs = self._xlabel_style.to_mpl_settings_dict() if self._xlabel_style else {}
            self._ax.set_xlabel(self.xlabel, **kwargs)

        if self.ylabel is not None:
            kwargs = self._ylabel_style.to_mpl_settings_dict() if self._ylabel_style else {}
            self._ax.set_ylabel(self.ylabel, **kwargs)

        if self.title is not None:
            kwargs = self._title_style.to_mpl_settings_dict() if self._title_style else {}
            self._ax.set_title(self.title, **kwargs)

    def set_xaxis_label_style(
        self,
        *,
        fontsize: float | int | None = None,
        fontweight: str | None = None,
        fontstyle: str | None = None,
        fontfamily: str | None = None,
        fontcolor: Color = "black",
        fontalpha: float | None = None,
        labelpad: float | None = None,
    ) -> None:
        """Sets the styling for the x-axis label.

        Kwargs:
            fontsize (float | int | None, optional): Font size for the x-axis label.
                Defaults to None.
            fontweight (str | None, optional): Font weight (e.g., "normal", "bold").
                Defaults to None.
            fontstyle (str | None, optional): Font style (e.g., "normal", "italic").
                Defaults to None.
            fontfamily (str | None, optional): Font family (e.g., "sans-serif", "serif").
                Defaults to None.
            fontcolor (Color, optional): Color of the x-axis label. Defaults to "black".
            fontalpha (float | None, optional): Alpha transparency of the x-axis label color.
                If None, uses alpha from color if specified. Defaults to None.
            labelpad (float | None, optional): Padding between the x-axis label and the axis
                in points. Defaults to None.

        Returns:
            None
        """
        self._xlabel_style = AxisLabelStyle(
            fontsize=fontsize,
            fontweight=fontweight,
            fontstyle=fontstyle,
            fontfamily=fontfamily,
            fontcolor=fontcolor,
            fontalpha=fontalpha,
            labelpad=labelpad,
        )

    def set_yaxis_label_style(
        self,
        *,
        fontsize: float | int | None = None,
        fontweight: str | None = None,
        fontstyle: str | None = None,
        fontfamily: str | None = None,
        fontcolor: Color = "black",
        fontalpha: float | None = None,
        labelpad: float | None = None,
    ) -> None:
        """Sets the styling for the y-axis label.

        Kwargs:
            fontsize (float | int | None, optional): Font size for the y-axis label.
                Defaults to None.
            fontweight (str | None, optional): Font weight (e.g., "normal", "bold").
                Defaults to None.
            fontstyle (str | None, optional): Font style (e.g., "normal", "italic").
                Defaults to None.
            fontfamily (str | None, optional): Font family (e.g., "sans-serif", "serif").
                Defaults to None.
            fontcolor (Color, optional): Color of the y-axis label. Defaults to "black".
            fontalpha (float | None, optional): Alpha transparency of the y-axis label color.
                If None, uses alpha from color if specified. Defaults to None.
            labelpad (float | None, optional): Padding between the y-axis label and the axis
                in points. Defaults to None.

        Returns:
            None
        """
        self._ylabel_style = AxisLabelStyle(
            fontsize=fontsize,
            fontweight=fontweight,
            fontstyle=fontstyle,
            fontfamily=fontfamily,
            fontcolor=fontcolor,
            fontalpha=fontalpha,
            labelpad=labelpad,
        )

    def set_title_style(
        self,
        *,
        fontsize: float | int | None = None,
        fontweight: str | None = None,
        fontstyle: str | None = None,
        fontfamily: str | None = None,
        fontcolor: Color = "black",
        fontalpha: float | None = None,
        loc: Literal["left", "center", "right"] | None = None,
        pad: float | None = None,
    ) -> None:
        """Sets the styling for the axes title.

        Kwargs:
            fontsize (float | int | None, optional): Font size for the title. Defaults to None.
            fontweight (str | None, optional): Font weight (e.g., "normal", "bold").
                Defaults to None.
            fontstyle (str | None, optional): Font style (e.g., "normal", "italic").
                Defaults to None.
            fontfamily (str | None, optional): Font family (e.g., "sans-serif", "serif").
                Defaults to None.
            fontcolor (Color, optional): Color of the title. Defaults to "black".
            fontalpha (float | None, optional): Alpha transparency of the title color.
                If None, uses alpha from color if specified. Defaults to None.
            loc (Literal["left", "center", "right"] | None, optional): Title location.
                Defaults to None.
            pad (float | None, optional): Padding between the title and the axes in points.
                Defaults to None.

        Returns:
            None
        """
        self._title_style = TitleStyle(
            fontsize=fontsize,
            fontweight=fontweight,
            fontstyle=fontstyle,
            fontfamily=fontfamily,
            fontcolor=fontcolor,
            fontalpha=fontalpha,
            loc=loc,
            pad=pad,
        )

    def set_xlabel(self, text: str | None) -> None:
        """Set the x-axis label text (deferred until build)."""
        self.xlabel = text

    def set_ylabel(self, text: str | None) -> None:
        """Set the y-axis label text (deferred until build)."""
        self.ylabel = text

    def set_title(self, text: str | None) -> None:
        """Set the axes title text (deferred until build)."""
        self.title = text

    def clear_xlabel_ylabel_and_title_styles(self) -> None:
        """Clear all xlabel/ylabel/title styles."""
        self._xlabel_style = None
        self._ylabel_style = None
        self._title_style = None

    def set_xaxis_tick_style(
        self,
        *,
        size: float | int = 10,
        rotation: float | int = 0,
        fontcolor: Color = "black",
        fontalpha: float | None = None,
        tickcolor: Color = "black",
        tickalpha: float | None = None,
        fontweight: str = "normal",
        fontstyle: str = "normal",
        fontfamily: str = "sans-serif",
        ticktype: TickType = "major",
    ) -> None:
        """Set x-axis tick style.

        Kwargs:
            size (float, optional): Font size of tick labels. Defaults to 10.
            rotation (float | int, optional): Rotation angle of tick labels in degrees.
                Defaults to 0.
            fontcolor (str, optional): Color of tick labels. Defaults to "black".
            fontalpha (float, optional): Alpha transparency of tick label color. If None,
                uses alpha from color if specified or will fall back to 1.0. Defaults to None.
            tickcolor (str, optional): Color of tick marks. Defaults to "black".
            tickalpha (float, optional): Alpha transparency of tick mark color. If None,
                uses alpha from color if specified or will fall back to 1.0. Defaults to None.
            fontweight (str, optional): Font weight of tick labels (e.g., 'normal 'bold').
                Defaults to "normal".
            fontstyle (str, optional): Font style of tick labels (e.g., 'normal', 'italic').
                Defaults to "normal".
            fontfamily (str, optional): Font family of tick labels (e.g., 'serif', 'sans-serif').
                Defaults to "sans-serif".
            ticktype (TickType, optional): Type of ticks to style ('major', 'minor', 'both').
                Defaults to 'major'.

        Returns:
            None
        """
        self._x_tick_style = TickStyle(
            size=size,
            rotation=rotation,
            fontcolor=fontcolor,
            fontalpha=fontalpha,
            tickcolor=tickcolor,
            tickalpha=tickalpha,
            fontweight=fontweight,
            fontstyle=fontstyle,
            fontfamily=fontfamily,
            ticktype=ticktype,
        )

    def set_yaxis_tick_style(
        self,
        *,
        size: float | int = 10,
        rotation: float | int = 0,
        fontcolor: Color = "black",
        fontalpha: float | None = None,
        tickcolor: Color = "black",
        tickalpha: float | None = None,
        fontweight: str = "normal",
        fontstyle: str = "normal",
        fontfamily: str = "sans-serif",
        ticktype: TickType = "major",
    ) -> None:
        """Set y-axis tick style.

        Kwargs:
            size (float, optional): Font size of tick labels. Defaults to 10.
            rotation (float | int, optional): Rotation angle of tick labels in degrees.
                Defaults to 0.
            fontcolor (str, optional): Color of tick labels. Defaults to "black".
            fontalpha (float, optional): Alpha transparency of tick label color. If None,
                uses alpha from color if specified or will fall back to 1.0. Defaults to None.
            tickcolor (str, optional): Color of tick marks. Defaults to "black".
            tickalpha (float, optional): Alpha transparency of tick mark color. If None,
                uses alpha from color if specified or will fall back to 1.0. Defaults to None.
            fontweight (str, optional): Font weight of tick labels (e.g., 'normal 'bold').
                Defaults to "normal".
            fontstyle (str, optional): Font style of tick labels (e.g., 'normal', 'italic').
                Defaults to "normal".
            fontfamily (str, optional): Font family of tick labels (e.g., 'serif', 'sans-serif').
                Defaults to "sans-serif".
            ticktype (TickType, optional): Type of ticks to style ('major', 'minor', 'both').
                Defaults to 'major'.

        Returns:
            None
        """
        self._y_tick_style = TickStyle(
            size=size,
            rotation=rotation,
            fontcolor=fontcolor,
            fontalpha=fontalpha,
            tickcolor=tickcolor,
            tickalpha=tickalpha,
            fontweight=fontweight,
            fontstyle=fontstyle,
            fontfamily=fontfamily,
            ticktype=ticktype,
        )

    def clear_xtick_labels(self) -> None:
        """Clear x-tick labels."""
        self._x_tick_labels = []

    def clear_ytick_labels(self) -> None:
        """Clear y-tick labels."""
        self._y_tick_labels = []

    def clear_xticks(self) -> None:
        """Clear x-tick locations and labels."""
        self._x_tick_locations = []
        self._x_tick_labels = []

    def clear_yticks(self) -> None:
        """Clear y-tick locations and labels."""
        self._y_tick_locations = []
        self._y_tick_labels = []

    def set_xlimits(self, lower: float, upper: float) -> None:
        """Set x-axis limits.

        Args:
            lower (float): The lower x-axis limit.
            upper (float): The upper x-axis limit.

        Returns:
            None
        """
        self._x_limits = (lower, upper)

    def set_ylimits(self, lower: float, upper: float) -> None:
        """Set y-axis limits.

        Args:
            lower (float): The lower y-axis limit.
            upper (float): The upper y-axis limit.

        Returns:
            None
        """
        self._y_limits = (lower, upper)

    def show_or_hide_frame(
        self,
        show_top: bool = True,
        show_right: bool = True,
        show_left: bool = True,
        show_bottom: bool = True,
    ) -> None:
        """Hide the frame of the plot.

        Args:
            show_top (bool, optional): Whether to show the top spine. Defaults to True.
            show_right (bool, optional): Whether to show the right spine. Defaults to True.
            show_left (bool, optional): Whether to show the left spine. Defaults to True.
            show_bottom (bool, optional): Whether to show the bottom spine. Defaults to True.

        Returns:
            None
        """
        self._frame_visibility = {
            "top": show_top,
            "right": show_right,
            "left": show_left,
            "bottom": show_bottom,
        }

    def _draw_verticals(self) -> None:
        """Draw vertical lines and bands on the plot."""
        for band in self._vertical_bands:
            edgecolor = (
                "none"
                if band.linecolor is None or band.linewidth == 0.0
                else mcolors.to_rgba(band.linecolor, alpha=band.linealpha)
            )
            self._ax.axvspan(
                band.lower_bound,
                band.upper_bound,
                facecolor=mcolors.to_rgba(band.bandcolor, alpha=band.bandalpha),
                edgecolor=edgecolor,
                linestyle=band.linestyle,
                linewidth=band.linewidth,
                zorder=band.zorder,
            )

        for ln in self._vertical_lines:
            vals = ln.values
            if isinstance(vals, Real) and not isinstance(vals, bool):
                vals = [float(vals)]
            for value in vals:
                self._ax.axvline(
                    value,
                    color=mcolors.to_rgba(ln.linecolor, alpha=ln.linealpha),
                    linestyle=ln.linestyle,
                    linewidth=ln.linewidth,
                    zorder=ln.zorder,
                )

    def _draw_horizontals(self) -> None:
        """Draw horizontal lines and bands on the plot."""
        for band in self._horizontal_bands:
            edgecolor = (
                "none"
                if band.linecolor is None or band.linewidth == 0.0
                else mcolors.to_rgba(band.linecolor, alpha=band.linealpha)
            )
            self._ax.axhspan(
                band.lower_bound,
                band.upper_bound,
                facecolor=mcolors.to_rgba(band.bandcolor, alpha=band.bandalpha),
                edgecolor=edgecolor,
                linestyle=band.linestyle,
                linewidth=band.linewidth,
                zorder=band.zorder,
            )

        for ln in self._horizontal_lines:
            vals = ln.values
            if isinstance(vals, Real) and not isinstance(vals, bool):
                vals = [float(vals)]
            for value in vals:
                self._ax.axhline(
                    value,
                    color=mcolors.to_rgba(ln.linecolor, alpha=ln.linealpha),
                    linestyle=ln.linestyle,
                    linewidth=ln.linewidth,
                    zorder=ln.zorder,
                )

    def _get_named_line_legend_handles(self) -> list[Any]:
        """Get legend handles for all named lines.

        Returns:
            list[Any]: A list of legend handles.
        """
        handles = []
        for line in self._vertical_lines + self._horizontal_lines:
            if line.name is not None:
                handle = Line2D(
                    [0],
                    [0],
                    color=mcolors.to_rgba(line.linecolor, alpha=line.linealpha),
                    linestyle=line.linestyle,
                    linewidth=line.linewidth,
                    label=line.name,
                )
                handles.append(handle)

        return handles

    def _get_named_band_legend_handles(self) -> list[Any]:
        """Get legend handles for all named bands.

        Returns:
            list[Any]: A list of legend handles.
        """
        handles = []
        for band in self._vertical_bands + self._horizontal_bands:
            if band.name is None:
                continue

            edgecolor = (
                "none"
                if band.linecolor is None or band.linewidth == 0.0
                else mcolors.to_rgba(band.linecolor, alpha=band.linealpha)
            )
            handle = Patch(
                facecolor=mcolors.to_rgba(band.bandcolor, alpha=band.bandalpha),
                edgecolor=edgecolor,
                linestyle=band.linestyle,
                linewidth=band.linewidth,
                label=band.name,
            )
            handles.append(handle)

        return handles

    def set_legend_options(
        self,
        *,
        loc: str | int = "center left",
        bbox_to_anchor: tuple[float, float] | tuple[float, float, float, float] | None = (
            1.01,
            0.5,
        ),
        ncols: int = 1,
        fontsize: float | str | None = None,
        frameon: bool = True,
        fancybox: bool = False,
        shadow: bool = False,
        framealpha: float | None = None,
        facecolor: Color | None = None,
        edgecolor: Color | None = None,
        title: str | None = None,
        alignment: Literal["center", "left", "right"] = "center",
        labelspacing: float = 0.5,
        columnspacing: float = 2.0,
    ) -> None:
        """Set legend options for the figure.

        This method stores legend placement and styling settings which are later passed to
        Matplotlib's ``Axes.legend(...)`` call when the plot is built. The defaults are chosen
        to place the legend just outside the axes on the right-hand side (useful for avoiding
        occluding plotted data).

        Notes:
            - Legend positioning is controlled by the interaction between ``loc`` and
              ``bbox_to_anchor``. Roughly: ``bbox_to_anchor`` specifies an anchor point/box,
              and ``loc`` specifies *which point on the legend* is aligned to that anchor.
            - The default ``loc="center_left"`` with ``bbox_to_anchor=(1.01, 0.5)`` anchors
              the legend's left edge slightly to the right of the axes (x=1.01) and vertically
              centers it (y=0.5).
            - ``bbox_to_anchor`` accepts either:
                * a 2-tuple ``(x, y)`` (anchor point), or
                * a 4-tuple ``(x, y, width, height)`` (anchor box).
              If ``bbox_to_anchor`` is ``None``, Matplotlib positions the legend using only ``loc``.
            - Spacing parameters such as ``labelspacing`` and ``columnspacing`` are interpreted
              by Matplotlib in *font-size units*, not pixels.

        Kwargs:
            loc (str | int, optional): The legend location. May be a Matplotlib string location
                (e.g., ``"best"``, ``"upper right"``) or an integer code. Defaults to
                ``"center_left"``.
            bbox_to_anchor (tuple[float, float] | tuple[float, float, float, float] | None, optional):
                The bounding box used to anchor the legend. A 2-tuple anchors to a single point;
                a 4-tuple anchors to a box. Defaults to ``(1.01, 0.5)``, which places the legend
                outside the axes to the right.
            ncols (int, optional): The number of columns in the legend. Increase this to make a
                tall legend more compact. Defaults to 1.
            fontsize (float | str | None, optional): The font size of the legend text. Accepts a
                numeric size (e.g. 10) or a Matplotlib size string (e.g. ``"small"``). If ``None``,
                Matplotlib's default is used. Defaults to None.
            frameon (bool, optional): Whether to draw a frame (background patch) around the legend.
                Defaults to True.
            fancybox (bool, optional): Whether to draw the legend frame with rounded corners.
                Defaults to False.
            shadow (bool, optional): Whether to draw a shadow behind the legend frame.
                Defaults to False.
            framealpha (float | None, optional): The alpha transparency of the legend frame. If ``None``,
                Matplotlib uses its default frame alpha. Defaults to None.
            facecolor (Color | None, optional): The face color of the legend frame. If ``None``,
                Matplotlib chooses a default. Defaults to None.
            edgecolor (Color | None, optional): The edge color of the legend frame. If ``None``,
                Matplotlib chooses a default. Defaults to None.
            title (str | None, optional): The title text displayed above legend entries. Defaults to None.
            alignment (Literal["center", "left", "right"], optional): The alignment of the legend title.
                Defaults to "center".
            labelspacing (float, optional): The vertical space between legend entries (in font-size units).
                Defaults to 0.5.
            columnspacing (float, optional): The horizontal space between columns (in font-size units).
                Defaults to 2.0.

        Examples:
            Place the legend inside the axes in the upper-right corner::

                plot.set_legend_options(loc="upper right", bbox_to_anchor=None)

            Place the legend below the axes, centered, with two columns::

                plot.set_legend_options(
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.05),
                    ncols=2,
                    frameon=False,
                )
        """
        self._legend_options = LegendOptions(
            loc=loc,
            bbox_to_anchor=bbox_to_anchor,
            ncols=ncols,
            fontsize=fontsize,
            frameon=frameon,
            fancybox=fancybox,
            shadow=shadow,
            framealpha=framealpha,
            facecolor=facecolor,
            edgecolor=edgecolor,
            title=title,
            alignment=alignment,
            labelspacing=labelspacing,
            columnspacing=columnspacing,
        )

    def save_legend(
        self,
        filepath: str,
        *,
        outer_padding: float = 0.07,
        dpi: int | None = None,
        **legend_kwargs: Any,
    ) -> None:
        """Save the legend to a separate file.

        Args:
            filepath (str): The file path to save the legend to.

        Kwargs:
            outer_padding (float, optional): The outer padding around the legend.
                Defaults to 0.07.

            Additional keyword arguments to pass to ``ax.legend()``.

        Returns:
            None
        """
        if not self._legend_handles:
            raise ValueError("No legend handles to save.")

        legend_fig = plt.figure(dpi=dpi or self.fig.dpi)
        legend_ax = legend_fig.add_subplot(111)
        legend_ax.axis("off")

        legend_options = self._legend_options.to_dict() | legend_kwargs

        leg = legend_ax.legend(
            handles=self._legend_handles,
            **legend_options,
        )

        # Make layout as tight as possible
        legend_fig.subplots_adjust(0, 0, 1, 1)

        legend_fig.canvas.draw()
        renderer = legend_fig.canvas.get_renderer()  # type: ignore[attr-defined]

        bbox = leg.get_window_extent(renderer=renderer)
        bbox_inches = bbox.transformed(legend_fig.dpi_scale_trans.inverted())
        bbox_inches = bbox_inches.expanded(1.0 + outer_padding, 1.0 + outer_padding)

        legend_fig.savefig(
            filepath,
            bbox_inches=bbox_inches,
            pad_inches=0.0,
        )
        plt.close(legend_fig)

    def _update_legend(self) -> None:
        """Update the legend on the plot."""
        if not self._legend_handles:
            return

        self._ax.legend(handles=self._legend_handles, **(self._legend_options.to_dict()))

    def _apply_frame_visibility(self) -> None:
        """Apply frame visibility settings to the axes."""
        if self._frame_visibility is None:
            return

        for spine, visible in self._frame_visibility.items():
            self._ax.spines[spine].set_visible(visible)

    def _build_and_apply_settings(self) -> None:
        """Build the plot and apply all settings."""
        self._ax.clear()
        self._build_plot()
        self._draw_verticals()
        self._draw_horizontals()
        self._set_x_axis()
        self._set_y_axis()
        self._apply_frame_visibility()
        self._apply_deferred_tick_styles()
        self._apply_deferred_label_styles()
        if self.include_legend:
            self._update_legend()

    @property
    def ax(self) -> Axes:
        """Build the plot by applying all settings and drawing elements.

        Returns:
            Axes: The matplotlib Axes object with all of the settings applied.
        """
        self._build_and_apply_settings()
        return self._ax

    def show(self) -> None:
        """Display the figure."""
        self._build_and_apply_settings()
        plt.show()

    def save(self, filepath: str, **kwargs) -> None:
        """Save the figure to a file.

        Args:
            filepath (str): The file path to save the figure to.

        Kwargs:
            Additional keyword arguments to pass to ``fig.savefig()``.

        Returns:
            None
        """
        self._build_and_apply_settings()

        kwargs["bbox_inches"] = kwargs.get("bbox_inches", "tight")
        kwargs["dpi"] = kwargs.get("dpi", self.fig.dpi)

        self.fig.savefig(filepath, **kwargs)

    @abstractmethod
    def _build_plot(self) -> None:
        """Build the plot by applying all settings and drawing elements."""
        pass

    @property
    @abstractmethod
    def _legend_handles(self) -> list[Any]:
        """Get legend handles for all named elements in the plot.

        Returns:
            list[Any]: A list of legend handles.
        """
        pass
