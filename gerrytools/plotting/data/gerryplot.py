import weakref
from abc import ABC, abstractmethod
from numbers import Real
from typing import Any, Iterable, Literal, Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from gerrytools.colors import resolve_color_and_alpha
from gerrytools.logging import get_logger
from gerrytools.plotting._figure_io import save_figure, show_figure
from gerrytools.plotting._legend_utils import build_legend_options, save_legend_handles
from gerrytools.plotting.data._gerryplot_dataclasses import BandData, LineData
from gerrytools.plotting.mpl.axis_title_style import AxisLabelStyle, TitleStyle
from gerrytools.plotting.mpl.tick_style import TickStyle
from gerrytools.plotting.utils import _coerce_real_iter
from gerrytools.typing import Color, TickType

logger = get_logger(__name__)


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
            include_legend (bool, optional): Whether to include a legend in the plot.
                Defaults to True.
            xlabel (str | None, optional): The label for the x-axis. Defaults to None.
            ylabel (str | None, optional): The label for the y-axis. Defaults to None.
            title (str | None, optional): The title of the plot. Defaults to None.
        """
        self.fig, self._ax = plt.subplots(figsize=figure_size, dpi=dpi)

        # IMPORTANT: prevent implicit display in notebooks
        # Only close in Jupyter so init doesn't display
        try:
            from IPython import get_ipython

            ip = get_ipython()
            if ip is not None and getattr(ip, "kernel", None) is not None:
                plt.close(self.fig)
        except Exception:
            pass

        self.include_legend = include_legend
        self._legend_options = build_legend_options()

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
        """Get subclass-provided default x-tick locations.

        Returns:
            list[float] | None: Default x-tick locations, or None to keep Matplotlib defaults.
        """
        return None

    def _default_x_tick_labels(self, tick_locations: list[float]) -> list[str] | None:
        """Get subclass-provided default x-tick labels.

        Args:
            tick_locations (list[float]): Final x-tick locations selected for the axes.

        Returns:
            list[str] | None: Tick labels aligned to ``tick_locations``, or None to keep current
                labels.
        """
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

        Args:
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

        Args:
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
            labels (Any): Iterable of Matplotlib ``Text`` tick-label objects.
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

    def _resolved_rgba(
        self,
        color: Color,
        alpha: float | None = None,
        *,
        field: str = "color",
    ) -> tuple[float, float, float, float]:
        """Resolve a ``Color`` plus optional alpha override to an RGBA tuple.

        Args:
            color (Color): GerryTools color input.
            alpha (float | None, optional): Optional alpha override. Defaults to None.
            field (str, optional): Field name used in validation and warning messages.
                Defaults to ``"color"``.

        Returns:
            tuple[float, float, float, float]: Resolved RGBA values in ``[0, 1]``.
        """
        resolved_color, resolved_alpha = resolve_color_and_alpha(
            color,
            alpha=alpha,
            allow_none=True,
            field=field,
            owner=self.__class__.__name__,
            logger=logger,
        )
        rgba = mcolors.to_rgba(resolved_color, alpha=resolved_alpha)
        return (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]))

    def _apply_tick_style(self, axis: Literal["x", "y", "both"], style: TickStyle) -> None:
        """Apply tick style to the specified axis.

        Args:
            axis (Literal["x", "y", "both"]): The axis to apply the style to.
            style (TickStyle): The tick style to apply.

        Returns:
            None
        """
        # Tick marks + tick label basics
        label_color_resolved = self._resolved_rgba(
            style.fontcolor,
            style.fontalpha,
            field="fontcolor",
        )
        tick_color_resolved = self._resolved_rgba(
            style.tickcolor,
            style.tickalpha,
            field="tickcolor",
        )
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

        Args:
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

        Args:
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

        Args:
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
        """Set deferred x-axis label text.

        Args:
            text (str | None): Label text, or None to clear.

        Returns:
            None
        """
        self.xlabel = text

    def set_ylabel(self, text: str | None) -> None:
        """Set deferred y-axis label text.

        Args:
            text (str | None): Label text, or None to clear.

        Returns:
            None
        """
        self.ylabel = text

    def set_title(self, text: str | None) -> None:
        """Set deferred axes title text.

        Args:
            text (str | None): Title text, or None to clear.

        Returns:
            None
        """
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

        Args:
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

        Args:
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

    def set_xticks(
        self,
        locations: Sequence[float] | None = None,
        *,
        labels: Sequence[str] | None = None,
    ) -> None:
        """Set x-axis tick locations and optionally labels.

        Args:
            locations (Sequence[float] | None, optional): X-axis tick locations. Defaults to None.
            labels (Sequence[str] | None, optional): X-axis tick labels. Defaults to None.

        Returns:
            None
        """
        self.update_xtick_values(
            locations=None if locations is None else list(locations),
            labels=None if labels is None else list(labels),
        )

    def set_yticks(
        self,
        locations: Sequence[float] | None = None,
        *,
        labels: Sequence[str] | None = None,
    ) -> None:
        """Set y-axis tick locations and optionally labels.

        Args:
            locations (Sequence[float] | None, optional): Y-axis tick locations. Defaults to None.
            labels (Sequence[str] | None, optional): Y-axis tick labels. Defaults to None.

        Returns:
            None
        """
        self.update_ytick_values(
            locations=None if locations is None else list(locations),
            labels=None if labels is None else list(labels),
        )

    def set_xlimits(self, lower: float, upper: float) -> None:
        """Set x-axis limits.

        Args:
            lower (float): Left x-axis limit.
            upper (float): Right x-axis limit.

        Returns:
            None
        """
        self._x_limits = (float(lower), float(upper))

    def set_ylimits(self, lower: float, upper: float) -> None:
        """Set y-axis limits.

        Args:
            lower (float): Bottom y-axis limit.
            upper (float): Top y-axis limit.

        Returns:
            None
        """
        self._y_limits = (float(lower), float(upper))

    def set_xlim(self, left: float, right: float) -> None:
        """Alias for :meth:`set_xlimits`.

        Args:
            left (float): Left x-axis limit.
            right (float): Right x-axis limit.

        Returns:
            None
        """
        self.set_xlimits(left, right)

    def set_ylim(self, bottom: float, top: float) -> None:
        """Alias for :meth:`set_ylimits`.

        Args:
            bottom (float): Bottom y-axis limit.
            top (float): Top y-axis limit.

        Returns:
            None
        """
        self.set_ylimits(bottom, top)

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
            if band.linecolor is None or band.linewidth == 0.0:
                edgecolor = "none"
            else:
                edgecolor = self._resolved_rgba(
                    band.linecolor,
                    band.linealpha,
                    field="linecolor",
                )
            self._ax.axvspan(
                band.lower_bound,
                band.upper_bound,
                facecolor=self._resolved_rgba(
                    band.bandcolor,
                    band.bandalpha,
                    field="bandcolor",
                ),
                edgecolor=edgecolor,
                linestyle=band.linestyle,
                linewidth=band.linewidth,
                zorder=band.zorder,
            )

        for ln in self._vertical_lines:
            vals = ln.values
            if isinstance(vals, Real) and not isinstance(vals, bool):
                vals = [float(vals)]
            assert isinstance(vals, Iterable)
            for value in vals:
                assert isinstance(value, (int, float))
                self._ax.axvline(
                    value,
                    color=self._resolved_rgba(
                        ln.linecolor,
                        ln.linealpha,
                        field="linecolor",
                    ),
                    linestyle=ln.linestyle,
                    linewidth=ln.linewidth,
                    zorder=ln.zorder,
                )

    def _draw_horizontals(self) -> None:
        """Draw horizontal lines and bands on the plot."""
        for band in self._horizontal_bands:
            if band.linecolor is None or band.linewidth == 0.0:
                edgecolor = "none"
            else:
                edgecolor = self._resolved_rgba(
                    band.linecolor,
                    band.linealpha,
                    field="linecolor",
                )
            self._ax.axhspan(
                band.lower_bound,
                band.upper_bound,
                facecolor=self._resolved_rgba(
                    band.bandcolor,
                    band.bandalpha,
                    field="bandcolor",
                ),
                edgecolor=edgecolor,
                linestyle=band.linestyle,
                linewidth=band.linewidth,
                zorder=band.zorder,
            )

        for ln in self._horizontal_lines:
            vals = ln.values
            if isinstance(vals, Real) and not isinstance(vals, bool):
                vals = [float(vals)]
            assert isinstance(vals, Iterable)
            for value in vals:
                assert isinstance(value, (int, float))
                self._ax.axhline(
                    value,
                    color=self._resolved_rgba(
                        ln.linecolor,
                        ln.linealpha,
                        field="linecolor",
                    ),
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
                    color=self._resolved_rgba(
                        line.linecolor,
                        line.linealpha,
                        field="linecolor",
                    ),
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

            if band.linecolor is None or band.linewidth == 0.0:
                edgecolor = "none"
            else:
                edgecolor = self._resolved_rgba(
                    band.linecolor,
                    band.linealpha,
                    field="linecolor",
                )
            handle = Patch(
                facecolor=self._resolved_rgba(
                    band.bandcolor,
                    band.bandalpha,
                    field="bandcolor",
                ),
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
        """Set legend options used by ``Axes.legend`` during plot build.

        Args:
            loc (str | int, optional): Matplotlib legend location. Defaults to
                ``"center left"``.
            bbox_to_anchor (tuple[float, float] | tuple[float, float, float, float] | None,
                optional): Legend anchor box. Defaults to ``(1.01, 0.5)``.
            ncols (int, optional): Number of legend columns. Defaults to ``1``.
            fontsize (float | str | None, optional): Legend text size. Defaults to None.
            frameon (bool, optional): Whether to draw the legend frame. Defaults to True.
            fancybox (bool, optional): Whether to use a rounded frame. Defaults to False.
            shadow (bool, optional): Whether to draw a shadow. Defaults to False.
            framealpha (float | None, optional): Frame alpha override. Defaults to None.
            facecolor (Color | None, optional): Frame face color. Defaults to None.
            edgecolor (Color | None, optional): Frame edge color. Defaults to None.
            title (str | None, optional): Legend title. Defaults to None.
            alignment (Literal["center", "left", "right"], optional): Legend content
                alignment. Defaults to ``"center"``.
            labelspacing (float, optional): Vertical spacing between entries.
                Defaults to ``0.5``.
            columnspacing (float, optional): Horizontal spacing between columns.
                Defaults to ``2.0``.

        Returns:
            None
        """
        self._legend_options = build_legend_options(
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
        """Save legend handles to a standalone image.

        Args:
            filepath (str): Output file path.
            outer_padding (float, optional): Fractional padding around the legend bounding box.
                Defaults to ``0.07``.
            dpi (int | None, optional): Output DPI. If None, uses figure DPI.
                Defaults to None.
            **legend_kwargs (Any): Additional keyword arguments passed to
                ``matplotlib.axes.Axes.legend``.

        Returns:
            None
        """
        save_legend_handles(
            handles=self._legend_handles,
            legend_options=self._legend_options,
            filepath=filepath,
            outer_padding=outer_padding,
            dpi=dpi or self.fig.dpi,
            **legend_kwargs,
        )

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
        show_figure(
            self.fig,
            non_gui_filename="gerrytools_plot.png",
            non_gui_prefix="GerryTools Plotting",
        )

    def save(self, filepath: str, **kwargs) -> None:
        """Save the figure to a file.

        Args:
            filepath (str): The file path to save the figure to.
            **kwargs (Any): Additional keyword arguments passed to ``Figure.savefig``.

        Returns:
            None
        """
        self._build_and_apply_settings()
        save_figure(self.fig, filepath, **kwargs)

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
        return []
