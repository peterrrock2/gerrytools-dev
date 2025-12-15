import logging
import math
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from gerrytools.logging import get_logger
from gerrytools.plotting.colors import HEX8_OR_NONE_PATTERN, convert_color_to_hexa_or_none
from gerrytools.typing import Color

logger = get_logger(__name__)


@dataclass(frozen=True)
class ScatterPointSettings:
    markerfacecolor: Color = "none"
    markerfacealpha: float | None = None
    marker: str = "o"
    markersize: float = 6.0
    markeredgecolor: Color = "black"
    markeredgealpha: float | None = None
    markeredgewidth: float = 0.6
    zorder: int = 4

    def __post_init__(self) -> None:
        new_color = convert_color_to_hexa_or_none(self.markerfacecolor)
        if HEX8_OR_NONE_PATTERN.match(new_color) is None:
            raise ValueError(f"Invalid color after conversion: {new_color!r}")

        if new_color.lower() == "none":
            object.__setattr__(self, "markerfacecolor", new_color)
            object.__setattr__(self, "markerfacealpha", 0.0)

        else:
            hex_color, alpha_from_color = new_color[:7], int(new_color[7:], 16) / 255.0
            object.__setattr__(self, "markerfacecolor", hex_color)

            old_alpha: float | None = (
                float(self.markerfacealpha) if self.markerfacealpha is not None else None
            )
            if old_alpha is not None and not (0.0 <= old_alpha <= 1.0):
                raise ValueError("Alpha must be between 0.0 and 1.0")

            if old_alpha is not None and old_alpha != alpha_from_color:
                logger.log(
                    level=logging.DEBUG,
                    msg=(
                        f"In ScatterPointSettings ignoring alpha from color {new_color} "
                        f"because explicit alpha {old_alpha} was provided."
                    ),
                )

            if old_alpha is None:
                object.__setattr__(self, "markerfacealpha", alpha_from_color)
            else:
                object.__setattr__(self, "markerfacealpha", old_alpha)

        lw = float(self.markeredgewidth)
        if not math.isfinite(lw):
            raise ValueError("markeredgewidth must be finite")
        if lw < 0:
            raise ValueError("markeredgewidth must be nonnegative")
        object.__setattr__(self, "markeredgewidth", lw)

        new_edge_color = convert_color_to_hexa_or_none(self.markeredgecolor)
        if HEX8_OR_NONE_PATTERN.match(new_edge_color) is None:
            raise ValueError(f"Invalid color after conversion: {new_edge_color!r}")

        if new_edge_color.lower() == "none":
            object.__setattr__(self, "markeredgecolor", new_edge_color)
            object.__setattr__(self, "markeredgealpha", 0.0)
        else:
            object.__setattr__(self, "markeredgecolor", new_edge_color[:7])
            edge_alpha_from_color = int(new_edge_color[7:], 16) / 255.0
            if self.markeredgealpha is not None:
                old_edge_alpha = float(self.markeredgealpha)
                if not (0.0 <= old_edge_alpha <= 1.0):
                    raise ValueError("markeredgealpha must be between 0.0 and 1.0")
                if old_edge_alpha != edge_alpha_from_color:
                    logger.log(
                        level=logging.DEBUG,
                        msg=(
                            f"In ScatterPointSettings ignoring alpha from edge color {new_edge_color} "
                            f"because explicit markeredgealpha {old_edge_alpha} was provided."
                        ),
                    )
                object.__setattr__(self, "markeredgealpha", old_edge_alpha)
            else:
                object.__setattr__(self, "markeredgealpha", edge_alpha_from_color)

        s = float(self.markersize)
        if not math.isfinite(s):
            raise ValueError("markersize must be finite")
        if s < 0:
            raise ValueError("markersize must be nonnegative")
        object.__setattr__(self, "markersize", s)

    def to_mpl_settings_dict(self) -> dict[str, Any]:
        """Convert the ScatterPointSettings to a dictionary.

        Returns:
            dict[str, Any]: A dictionary representation of the ScatterPointSettings.
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
class LineData:
    value: float
    linecolor: Color = "#cccccc"
    linealpha: float | None = None
    linestyle: str = "-"
    linewidth: float = 1.0
    zorder: int = -1
    name: str | None = None

    def __post_init__(self) -> None:
        hex8 = convert_color_to_hexa_or_none(self.linecolor)
        if HEX8_OR_NONE_PATTERN.match(hex8) is None:
            raise ValueError(f"Line color {self.linecolor} could not be converted to valid color.")

        if hex8.lower() == "none":
            raise ValueError("LineData.linecolor cannot be 'none'.")

        object.__setattr__(self, "linecolor", hex8[:7])
        alpha_from_color = int(hex8[7:], 16) / 255.0

        old_alpha: float | None = float(self.linealpha) if self.linealpha is not None else None
        if old_alpha is not None and not (0.0 <= old_alpha <= 1.0):
            raise ValueError("Alpha must be between 0.0 and 1.0")

        if old_alpha is not None and old_alpha != alpha_from_color:
            logger.log(
                level=logging.DEBUG,
                msg=(
                    f"For LineData {self.name}: Ignoring alpha from color {hex8} "
                    f"because explicit alpha {old_alpha} was provided."
                ),
            )
        object.__setattr__(
            self,
            "linealpha",
            old_alpha if old_alpha is not None else alpha_from_color,
        )
        lw = float(self.linewidth)
        if lw < 0:
            raise ValueError("LineData.linewidth must be nonnegative.")
        if not math.isfinite(lw):
            raise ValueError("LineData.linewidth must be finite.")
        object.__setattr__(self, "linewidth", lw)

        object.__setattr__(self, "zorder", int(self.zorder))


@dataclass(frozen=True)
class BandData:
    lower_bound: float
    upper_bound: float
    bandcolor: Color = "#cccccc"
    alpha: float | None = None
    linecolor: Color | None = None
    linealpha: float | None = None
    linestyle: str = "-"
    linewidth: float = 1.0
    zorder: int = -1
    name: str | None = None

    def __post_init__(self) -> None:
        old_alpha: float | None = float(self.alpha) if self.alpha is not None else None
        if old_alpha is not None and not (0.0 <= old_alpha <= 1.0):
            raise ValueError("Alpha must be between 0.0 and 1.0")

        bandcolor = self.bandcolor
        if isinstance(bandcolor, str) and bandcolor.lower() == "none":
            bandcolor = None

        original_bandcolor = bandcolor
        bandalpha_from_color: float | None = None

        if bandcolor is not None:
            band_hex8 = convert_color_to_hexa_or_none(bandcolor)
            if HEX8_OR_NONE_PATTERN.match(band_hex8) is None:
                raise ValueError(
                    f"Band color {bandcolor} could not be converted to valid color got {band_hex8}."
                )

            bandcolor = band_hex8[:7]
            bandalpha_from_color = int(band_hex8[7:], 16) / 255.0

        object.__setattr__(self, "bandcolor", bandcolor)

        if (
            old_alpha is not None
            and bandalpha_from_color is not None
            and old_alpha != bandalpha_from_color
        ):
            logger.log(
                level=logging.DEBUG,
                msg=(
                    f"For BandData {self.name}: Ignoring alpha from color {original_bandcolor} "
                    f"because explicit alpha {old_alpha} was provided."
                ),
            )

        if old_alpha is None:
            object.__setattr__(self, "alpha", bandalpha_from_color)
        else:
            object.__setattr__(self, "alpha", old_alpha)

        lb, ub = sorted([self.lower_bound, self.upper_bound])

        object.__setattr__(self, "lower_bound", lb)
        object.__setattr__(self, "upper_bound", ub)
        linecolor_orig = self.linecolor
        if linecolor_orig is None:
            fallback_linecolor = self.bandcolor or "#cccccc"
            linecolor_orig = fallback_linecolor

        # treat "none" as "no border"
        if isinstance(linecolor_orig, str) and linecolor_orig.lower() == "none":
            object.__setattr__(self, "linecolor", "none")
            object.__setattr__(self, "linealpha", None)
            object.__setattr__(self, "linewidth", 0.0)
            object.__setattr__(self, "zorder", int(self.zorder))
            return

        normalized_line = LineData(
            value=lb,
            linecolor=linecolor_orig,
            linealpha=self.linealpha,
            linestyle=self.linestyle,
            linewidth=self.linewidth,
            zorder=self.zorder,
        )

        object.__setattr__(self, "linecolor", normalized_line.linecolor)
        object.__setattr__(self, "linealpha", normalized_line.linealpha)
        object.__setattr__(self, "linestyle", normalized_line.linestyle)
        object.__setattr__(self, "linewidth", normalized_line.linewidth)
        object.__setattr__(self, "zorder", normalized_line.zorder)


class GerryPlotBase(ABC):
    def __init__(
        self,
        figure_size: tuple[float, float] = (10, 6),
        dpi: int = 300,
        *,
        include_legend: bool = False,
    ) -> None:
        """Initialize a GerryPlotBase instance.

        Args:
            figure_size (tuple[float, float], optional): The size of the figure in inches.
                Defaults to (10, 6).
            dpi (int, optional): The dots per inch (DPI) of the figure. Defaults to 300.

        """
        self.fig, self.ax = plt.subplots(figsize=figure_size, dpi=dpi)

        self.include_legend = include_legend
        self.legend_loc = "center left"
        self.legend_bbox_to_anchor = (1.01, 0.5)

        self._x_tick_locations: list[float] | None = None
        self._x_tick_labels: list[str] | None = None
        self._x_limits: tuple[float, float] | None = None

        self._y_tick_locations: list[float] | None = None
        self._y_tick_labels: list[str] | None = None
        self._y_limits: tuple[float, float] | None = None

        self._vertical_lines: list[LineData] = []
        self._vertical_bands: list[BandData] = []
        self._horizontal_lines: list[LineData] = []
        self._horizontal_bands: list[BandData] = []

        self._finalizer = weakref.finalize(self, plt.close, self.fig)

    def add_vertical_line(
        self,
        x_value: float,
        *,
        linecolor: Color = "#cccccc",
        linealpha: float | None = None,
        linestyle: str = "-",
        linewidth: float = 1.0,
        zorder: int = -1,
        name: str | None = None,
    ) -> None:
        """Add a vertical line to the figure.

        Args:
            x_value (float): The x-value where the vertical line should be drawn.

        Kwargs:
            linecolor (Color, optional): The color of the vertical line. Defaults to "#cccccc".
            linestyle (str, optional): The linestyle of the vertical line. Defaults to "-".
            linewidth (float, optional): The width of the vertical line. Defaults to 1.0.
            zorder (int, optional): The z-order of the vertical line. Defaults to -1.
            name (str | None, optional): The name of the line for legend purposes. Defaults to None.

        Returns:
            None
        """
        self._vertical_lines.append(
            LineData(
                value=float(x_value),
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
        alpha: float | None = None,
        linecolor: Color | None = None,
        linealpha: float | None = None,
        linestyle: str = "-",
        linewidth: float = 1.0,
        zorder: int = -1,
        name: str | None = None,
    ) -> None:
        """Add a vertical band to the figure.

        Args:
            x_low (float): The lower x-value of the vertical band.
            x_high (float): The upper x-value of the vertical band.

        Kwargs:
            bandcolor (Color | None, optional): The fill color of the band. Defaults to "#cccccc".
            alpha (float | None, optional): The alpha transparency of the band. Defaults to None.
            linecolor (Color | None, optional): The color of the bounding lines of the band.
                If set to None and bandcolor is also None, defaults to "#cccccc".
                If set to None and bandcolor is not None, defaults to bandcolor.
                Defaults to None.
            linestyle (str, optional): The linestyle of the bounding lines of the band.
                Defaults to "-".
            linewidth (float, optional): The width of the bounding lines of the band.
                Defaults to 1.0.
            zorder (int, optional): The z-order of the band. Defaults to -1.
            name (str | None, optional): The name of the band for legend purposes. Defaults to None.

        Returns:
            None
        """
        self._vertical_bands.append(
            BandData(
                lower_bound=min(x_low, x_high),
                upper_bound=max(x_low, x_high),
                bandcolor=bandcolor,
                alpha=alpha,
                linecolor=linecolor,
                linealpha=linealpha,
                linestyle=linestyle,
                linewidth=linewidth,
                zorder=zorder,
                name=name,
            )
        )

    def add_horizontal_line(
        self,
        y_value: float,
        *,
        linecolor: Color = "#cccccc",
        linealpha: float | None = None,
        linestyle: str = "-",
        linewidth: float = 1.0,
        zorder: int = -2,
        name: str | None = None,
    ) -> None:
        """Add a horizontal line to the figure.

        Args:
            y_value (float): The y-value where the horizontal line should be drawn.

        Kwargs:
            linecolor (Color, optional): The color of the horizontal line. Defaults to "#cccccc".
            linestyle (str, optional): The linestyle of the horizontal line. Defaults to "-".
            linewidth (float, optional): The width of the horizontal line. Defaults to 1.0.
            zorder (int, optional): The z-order of the horizontal line. Defaults to -2.
            name (str | None, optional): The name of the line for legend purposes. Defaults to None.

        Returns:
            None
        """
        self._horizontal_lines.append(
            LineData(
                value=float(y_value),
                linecolor=linecolor,
                linealpha=linealpha,
                linestyle=linestyle,
                linewidth=float(linewidth),
                zorder=zorder,
                name=name,
            )
        )

    def add_horizontal_band(
        self,
        y_low: float,
        y_high: float,
        *,
        bandcolor: Color = "#cccccc",
        alpha: float | None = None,
        linecolor: Color | None = None,
        linealpha: float | None = None,
        linestyle: str = "-",
        linewidth: float = 1.0,
        zorder: int = -2,
        name: str | None = None,
    ) -> None:
        """Add a horizontal band to the figure.

        Args:
            y_low (float): The lower y-value of the horizontal band.
            y_high (float): The upper y-value of the horizontal band.

        Kwargs:
            bandcolor (Color | None, optional): The fill color of the band. Defaults to "#cccccc".
            alpha (float | None, optional): The alpha transparency of the band. Defaults to None
            linecolor (Color | None, optional): The color of the bounding lines of the band.
                If set to None and bandcolor is also None, defaults to "#cccccc".
                If set to None and bandcolor is not None, defaults to bandcolor.
                Defaults to None.
            linestyle (str, optional): The linestyle of the bounding lines of the band.
                Defaults to "-".
            linewidth (float, optional): The width of the bounding lines of the band.
                Defaults to 1.0.
            zorder (int, optional): The z-order of the band. Defaults to -2.
            name (str | None, optional): The name of the band for legend purposes. Defaults to None.

        Returns:
            None
        """
        self._horizontal_bands.append(
            BandData(
                lower_bound=min(y_low, y_high),
                upper_bound=max(y_low, y_high),
                bandcolor=bandcolor,
                alpha=alpha,
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

    def _set_x_axis(self) -> None:
        """Set x-axis limits, ticks, and labels in the plot."""
        x_limits = self._x_limits if self._x_limits is not None else self.ax.get_xlim()
        self.ax.set_xlim(x_limits)

        if self._x_tick_locations is not None:
            x_tick_locations = list(self._x_tick_locations)
            self.ax.set_xticks(x_tick_locations)
        else:
            x_tick_locations = self.ax.get_xticks().tolist()

        if self._x_tick_labels is None:
            return

        if self._x_tick_labels == []:
            self.ax.set_xticks(ticks=x_tick_locations)
            self.ax.tick_params(axis="x", labelbottom=False)
            return

        if self._x_tick_locations is None:
            self.ax.set_xticks(x_tick_locations)

        if len(self._x_tick_labels) != len(x_tick_locations):
            raise ValueError(
                f"Expected {len(x_tick_locations)} x tick labels, got {len(self._x_tick_labels)}."
            )

        self.ax.set_xticklabels(list(self._x_tick_labels))

    def _set_y_axis(self) -> None:
        """Set y-axis limits, ticks, and labels in the plot."""
        y_limits = self._y_limits if self._y_limits is not None else self.ax.get_ylim()
        self.ax.set_ylim(y_limits)

        if self._y_tick_locations is not None:
            y_tick_locations = list(self._y_tick_locations)
            self.ax.set_yticks(y_tick_locations)
        else:
            y_tick_locations = self.ax.get_yticks().tolist()

        if self._y_tick_labels is None:
            return

        if self._y_tick_labels == []:
            self.ax.tick_params(axis="y", labelleft=False)
            return

        if self._y_tick_locations is None:
            self.ax.set_yticks(y_tick_locations)

        if len(self._y_tick_labels) != len(y_tick_locations):
            raise ValueError(
                f"Expected {len(y_tick_locations)} y tick labels, got {len(self._y_tick_labels)}."
            )

        self.ax.set_yticklabels(list(self._y_tick_labels))

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

    def set_xaxis_fontsize(self, size: float) -> None:
        """Set the font size of x-axis tick labels."""
        self.ax.tick_params(axis="x", labelsize=size)

    def set_yaxis_fontsize(self, size: float) -> None:
        """Set the font size of y-axis tick labels."""
        self.ax.tick_params(axis="y", labelsize=size)

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
        """Set x-axis limits."""
        self._x_limits = (lower, upper)

    def set_ylimits(self, lower: float, upper: float) -> None:
        """Set y-axis limits."""
        self._y_limits = (lower, upper)

    def hide_frame(
        self, top: bool = True, right: bool = True, left: bool = True, bottom: bool = True
    ) -> None:
        """Hide the frame of the plot.

        Kwargs:
            top (bool, optional): Whether to hide the top spine. Defaults to True.
            right (bool, optional): Whether to hide the right spine. Defaults to True.
            left (bool, optional): Whether to hide the left spine. Defaults to True.
            bottom (bool, optional): Whether to hide the bottom spine. Defaults to True.

        Returns:
            None
        """
        if top:
            self.ax.spines["top"].set_visible(False)
        if right:
            self.ax.spines["right"].set_visible(False)
        if left:
            self.ax.spines["left"].set_visible(False)
        if bottom:
            self.ax.spines["bottom"].set_visible(False)

    def _draw_verticals(self) -> None:
        """Draw vertical lines and bands on the plot."""
        for band in self._vertical_bands:
            edgecolor = (
                "none"
                if band.linecolor is None or band.linewidth == 0.0
                else mcolors.to_rgba(band.linecolor, alpha=band.linealpha)
            )
            self.ax.axvspan(
                band.lower_bound,
                band.upper_bound,
                facecolor=(
                    mcolors.to_rgba(band.bandcolor, alpha=band.alpha)
                    if band.bandcolor is not None
                    else "none"
                ),
                edgecolor=edgecolor,
                linestyle=band.linestyle,
                linewidth=band.linewidth,
                zorder=band.zorder,
            )

        for ln in self._vertical_lines:
            self.ax.axvline(
                ln.value,
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
            self.ax.axhspan(
                band.lower_bound,
                band.upper_bound,
                facecolor=(
                    mcolors.to_rgba(band.bandcolor, alpha=band.alpha)
                    if band.bandcolor is not None
                    else "none"
                ),
                edgecolor=edgecolor,
                linestyle=band.linestyle,
                linewidth=band.linewidth,
                zorder=band.zorder,
            )

        for ln in self._horizontal_lines:
            self.ax.axhline(
                ln.value,
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
                facecolor=(
                    "none"
                    if band.bandcolor is None
                    else mcolors.to_rgba(band.bandcolor, alpha=band.alpha)
                ),
                edgecolor=edgecolor,
                linestyle=band.linestyle,
                linewidth=band.linewidth,
                label=band.name,
            )
            handles.append(handle)

        return handles

    @property
    @abstractmethod
    def _legend_handles(self) -> list[Any]:
        """Get legend handles for all named elements in the plot.

        Returns:
            list[Any]: A list of legend handles.
        """
        pass

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

            Additional keyword arguments to pass to `ax.legend()`.

        Returns:
            None
        """
        if not self._legend_handles:
            raise ValueError("No legend handles to save.")

        legend_fig = plt.figure(dpi=dpi or self.fig.dpi)
        legend_ax = legend_fig.add_subplot(111)
        legend_ax.axis("off")

        leg = legend_ax.legend(
            handles=self._legend_handles,
            loc="center",
            frameon=True,
            **legend_kwargs,
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

    def show(self) -> None:
        """Display the figure."""
        self._build_plot()
        plt.show()

    def save(self, filepath: str, **kwargs) -> None:
        """Save the figure to a file.

        Args:
            filepath (str): The file path to save the figure to.

        Kwargs:
            Additional keyword arguments to pass to `fig.savefig()`.

        Returns:
            None
        """
        self._build_plot()
        kwargs["bbox_inches"] = kwargs.get("bbox_inches", "tight")
        kwargs["dpi"] = kwargs.get("dpi", self.fig.dpi)

        self.fig.savefig(filepath, **kwargs)

    @abstractmethod
    def _build_plot(self) -> Axes:
        """Build the plot by applying all settings and drawing elements."""
        pass
