from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from gerrytools.colors import resolve_color_and_alpha
from gerrytools.logging import get_logger
from gerrytools.plotting.data._geometry import line_segment_through_unit_square
from gerrytools.plotting.data.gerryplot import GerryPlotBase
from gerrytools.typing import Color, LegendHandle

logger = get_logger(__name__)


@dataclass(frozen=True)
class PaintBallLine:
    """Dataclass for storing paintball line properties.

    Attributes:
        slope (float): Slope of the guide line.
        linecolor (Color): Guide line color.
        linewidth (float): Guide line width.
        linestyle (str): Matplotlib line style string.
        linealpha (float | None): Optional alpha override for line color.
        zorder (int): Drawing order for the guide line.
        label (str | None): Optional legend label.
    """

    slope: float
    linecolor: Color
    linewidth: float
    linestyle: str
    linealpha: float | None = None
    zorder: int = -1
    label: str | None = None

    def __post_init__(self) -> None:
        slope = float(self.slope)
        if math.isnan(slope):
            raise ValueError("slope must not be NaN.")
        object.__setattr__(self, "slope", slope)

        line_width = float(self.linewidth)
        if not math.isfinite(line_width):
            raise ValueError("linewidth must be finite.")
        if line_width < 0:
            raise ValueError("linewidth must be nonnegative.")
        object.__setattr__(self, "linewidth", line_width)

        resolved_color, resolved_alpha = resolve_color_and_alpha(
            self.linecolor,
            alpha=self.linealpha,
            allow_none=True,
            field="linecolor",
            owner="PaintBallLine",
            logger=logger,
        )
        object.__setattr__(self, "linecolor", resolved_color)
        object.__setattr__(self, "linealpha", resolved_alpha)
        object.__setattr__(self, "zorder", int(self.zorder))


class PaintBall(GerryPlotBase):
    """A class for creating paintball plots in Matplotlib."""

    def __init__(
        self,
        voteshare_data: Iterable[float],
        seats_data: Iterable[float],
        maximum_seats: int | None = None,
        *,
        include_efficiency_gap_line: bool = True,
        include_proportionality_line: bool = True,
        figure_size: tuple[float, float] = (10, 10),
        dpi: int = 300,
        include_legend: bool = False,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
    ) -> None:
        """Initialize a paintball plot.

        Args:
            voteshare_data (Iterable[float]): Vote-share values for each plan outcome.
                Every value must be in the closed interval [0, 1].
            seats_data (Iterable[float]): Seat-share values or seat counts for each plan outcome.
                If ``maximum_seats`` is None, values are interpreted as seat shares and must be
                in [0, 1]. If ``maximum_seats`` is provided, values are interpreted as seat counts
                and are normalized by dividing by ``maximum_seats``.
            maximum_seats (int | None, optional): Maximum seat count used to normalize
                ``seats_data`` to share values when provided. Defaults to None.
            include_efficiency_gap_line (bool, optional): Whether to add the
                efficiency-gap guide line by default. Defaults to True.
            include_proportionality_line (bool, optional): Whether to add the
                proportionality guide line by default. Defaults to True.
            figure_size (tuple[float, float], optional): Figure size in inches.
                Defaults to (10, 10).
            dpi (int, optional): Figure DPI. Defaults to 300.
            include_legend (bool, optional): Whether to include legend when rendering.
                Defaults to False.
            xlabel (str | None, optional): X-axis label text. Defaults to None.
            ylabel (str | None, optional): Y-axis label text. Defaults to None.
            title (str | None, optional): Plot title text. Defaults to None.
        """
        super().__init__(
            figure_size=figure_size,
            dpi=dpi,
            include_legend=include_legend,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
        )

        self._voteshare_data, self._seatshare_data = (
            self._validate_voteshare_seatshare_and_max_seats(
                list(voteshare_data),
                list(seats_data),
                maximum_seats,
            )
        )

        self._named_lines: dict[str, PaintBallLine] = {}
        self._lines: dict[float, list[PaintBallLine]] = {}

        self._draw_hull = False

        self.clear_options()

        if include_efficiency_gap_line:
            self.add_lines_with_slope(
                slopes=[2.0],
                linecolor="gray",
                linewidth=1.0,
                linestyle="-",
                name="Efficiency Gap",
            )

        if include_proportionality_line:
            self.add_lines_with_slope(
                slopes=[1.0],
                linecolor="gray",
                linewidth=1.0,
                linestyle="--",
                name="Proportionality",
            )

    def clear_options(self) -> None:
        """Reset all display options to defaults."""
        self.markersize = 16.0
        self.marker = "o"
        self.markerfacecolor = "cadmiumgreen"
        self.markerfacealpha = 0.8
        self.markeredgecolor = "cadmiumgreen"
        self.markeredgewidth = 0.5
        self.markeredgealpha = 1.0

        self.hullcolor: Color | None = None
        self.hullalpha: float | None = None
        self.hulledgecolor: Color | None = None
        self.hulledgewidth: float = 2.0
        self.hulledgealpha: float | None = None

        self.crosshair_color: Color = "lightgrey"
        self.crosshair_width = 5.0
        self.crosshair_alpha = 1.0

        self.xscale = 10.0
        self.yscale = 10.0

        self.set_xlim(0.0, 1.0)
        self.set_ylim(0.0, 1.0)
        self.set_xticks(locations=[])
        self.set_yticks(locations=[])

    # ====================
    #   FEATURE ADDITION
    # ====================
    def _validate_voteshare_seatshare_and_max_seats(
        self,
        voteshare_data: list[float],
        seats_data: list[float],
        maximum_seats: int | None = None,
    ) -> tuple[list[float], list[float]]:
        """Validate and normalize incoming vote-share and seat-share data.

        ``voteshare_data`` values must be in [0, 1].
        ``seats_data`` is either interpreted directly as seat share values in [0, 1]
        (when ``maximum_seats`` is None), or as seat counts normalized by ``maximum_seats``
        (when ``maximum_seats`` is provided), and the resulting shares must also lie in [0, 1].

        Args:
            voteshare_data (list[float]): Vote-share values in ``[0, 1]``.
            seats_data (list[float]): Seat-share values in ``[0, 1]`` or raw seat counts.
            maximum_seats (int | None, optional): Total seats used to normalize raw seat counts.
                Defaults to None.

        Returns:
            tuple[list[float], list[float]]: Normalized vote-share and seat-share vectors.

        Raises:
            ValueError: If lengths mismatch, inputs are empty, or normalized values are out of
                range.
        """
        if len(voteshare_data) != len(seats_data):
            raise ValueError("voteshare_data and seats_data must have the same length.")
        if len(voteshare_data) == 0:
            raise ValueError("voteshare_data and seats_data must have at least one element.")

        ret_voteshare = [float(v) for v in voteshare_data]
        if not all(0.0 <= v <= 1.0 for v in ret_voteshare):
            raise ValueError("All vote-share values must be in [0, 1].")

        if maximum_seats is None:
            ret_seatshare = [float(s) for s in seats_data]
        else:
            if maximum_seats <= 0:
                raise ValueError("maximum_seats must be a positive integer when provided.")
            ret_seatshare = [float(s) / maximum_seats for s in seats_data]

        if not all(0.0 <= s <= 1.0 for s in ret_seatshare):
            raise ValueError("All seat-share values must be in [0, 1].")

        return ret_voteshare, ret_seatshare

    def add_voteshare_seatshare_data(
        self,
        voteshare_data: Iterable[float],
        seats_data: Iterable[float],
        maximum_seats: int | None = None,
    ) -> None:
        """Add vote-share / seat-share data points to the paintball plot.

        Args:
            voteshare_data (Iterable[float]): Vote-share values to add. Every value must be
                in [0, 1].
            seats_data (Iterable[float]): Seat-share values or seat counts to add.
                If ``maximum_seats`` is None, values are interpreted as seat shares and must be
                in [0, 1]. If ``maximum_seats`` is provided, values are interpreted as seat counts
                and are normalized by dividing by ``maximum_seats``.
            maximum_seats (int | None, optional): Maximum seat count used to normalize
                ``seats_data`` to share values when provided. Defaults to None.
        """
        new_voteshare_data, new_seatshare_data = self._validate_voteshare_seatshare_and_max_seats(
            list(voteshare_data),
            list(seats_data),
            maximum_seats,
        )
        self._voteshare_data.extend(new_voteshare_data)
        self._seatshare_data.extend(new_seatshare_data)

    def add_lines_with_slope(
        self,
        slopes: Iterable[float],
        linecolor: Color = "black",
        linewidth: float = 1.0,
        linestyle: str = "-",
        *,
        linealpha: float | None = None,
        zorder: int = -1,
        name: str | None = None,
    ) -> None:
        """Add guide lines with specified slopes.

        Args:
            slopes (Iterable[float]): Slopes for lines constrained to pass through (0.5, 0.5).
            linecolor (Color, optional): Line color. Defaults to "black".
            linealpha (float | None, optional): Line alpha override. Defaults to None.
            linewidth (float, optional): Line width. Defaults to 1.0.
            linestyle (str, optional): Matplotlib line style string. Defaults to "-".
            zorder (int, optional): Draw order for these lines. Defaults to -1.
            name (str | None, optional): If provided, all lines are stored as a named line for
                legend display. Defaults to None.
        """
        for slope in slopes:
            line = PaintBallLine(
                slope=float(slope),
                linecolor=linecolor,
                linealpha=linealpha,
                linewidth=linewidth,
                linestyle=linestyle,
                zorder=zorder,
                label=name,
            )
            if name is not None:
                self._named_lines[name] = line
            else:
                self._lines.setdefault(line.slope, []).append(line)

    def clear_lines(self) -> None:
        """Remove all custom and named lines from the paintball plot."""
        self._lines = {}
        self._named_lines = {}

    # ==================
    #   OPTION SETTERS
    # ==================
    def set_xlim(self, left: float, right: float) -> None:
        """Set x-axis limits for the paintball plot.

        Args:
            left (float): Lower x-axis limit.
            right (float): Upper x-axis limit.
        """
        if not (left < right):
            raise ValueError("left must be less than right.")

        self.set_xlimits(float(left), float(right))

    def set_ylim(self, bottom: float, top: float) -> None:
        """Set y-axis limits for the paintball plot.

        Args:
            bottom (float): Lower y-axis limit.
            top (float): Upper y-axis limit.
        """
        if not (bottom < top):
            raise ValueError("bottom must be less than top.")

        self.set_ylimits(float(bottom), float(top))

    def set_xscale(self, xscale: float) -> None:
        """Set horizontal scaling for the paintball plot.

        Args:
            xscale (float): Horizontal scaling factor.
        """
        xscale = float(xscale)
        if not math.isfinite(xscale):
            raise ValueError("xscale must be finite.")
        if xscale <= 0:
            raise ValueError("xscale must be positive.")
        self.xscale = xscale

    def set_yscale(self, yscale: float) -> None:
        """Set vertical scaling for the paintball plot.

        Args:
            yscale (float): Vertical scaling factor.
        """
        yscale = float(yscale)
        if not math.isfinite(yscale):
            raise ValueError("yscale must be finite.")
        if yscale <= 0:
            raise ValueError("yscale must be positive.")
        self.yscale = yscale

    def set_scale(self, xscale: float | None = None, yscale: float | None = None) -> None:
        """Set horizontal and/or vertical scaling for the paintball plot.

        Args:
            xscale (float | None, optional): Horizontal scaling factor. Defaults to None.
            yscale (float | None, optional): Vertical scaling factor. Defaults to None.
        """
        if xscale is not None:
            self.set_xscale(xscale)
        if yscale is not None:
            self.set_yscale(yscale)

    def set_crosshair_options(
        self,
        color: Color | None = None,
        width: float | None = None,
        *,
        alpha: float | None = None,
    ) -> None:
        """Set crosshair display options.

        Args:
            color (Color | None, optional): Crosshair color. Defaults to None.
            width (float | None, optional): Crosshair line width. Defaults to None.
            alpha (float | None, optional): Crosshair alpha in [0, 1]. Defaults to None.
        """
        if color is not None:
            self.crosshair_color = color
        if width is not None:
            width = float(width)
            if not math.isfinite(width):
                raise ValueError("crosshair width must be finite.")
            if width < 0:
                raise ValueError("crosshair width must be nonnegative.")
            self.crosshair_width = width
        if alpha is not None:
            if not (0.0 <= alpha <= 1.0):
                raise ValueError("alpha must be in [0, 1].")
            self.crosshair_alpha = float(alpha)

    def set_marker_options(
        self,
        size: float | None = None,
        color: Color | None = None,
        alpha: float | None = None,
        edgecolor: Color | None = None,
        edgewidth: float | None = None,
        edgealpha: float | None = None,
        *,
        marker: str | None = None,
    ) -> None:
        """Set marker display options.

        Args:
            size (float | None, optional): Marker size in points. Defaults to None.
            color (Color | None, optional): Marker face color. Defaults to None.
            alpha (float | None, optional): Marker face alpha in [0, 1]. Defaults to None.
            edgecolor (Color | None, optional): Marker edge color. Defaults to None.
            edgewidth (float | None, optional): Marker edge width. Defaults to None.
            edgealpha (float | None, optional): Marker edge alpha in [0, 1]. Defaults to None.
            marker (str | None, optional): Matplotlib marker style string. Defaults to None.
        """
        if size is not None:
            size = float(size)
            if not math.isfinite(size):
                raise ValueError("size must be finite.")
            if size <= 0:
                raise ValueError("size must be positive.")
            self.markersize = size

        if marker is not None:
            self.marker = str(marker)

        if color is not None:
            self.markerfacecolor = color

        if alpha is not None:
            if not (0.0 <= alpha <= 1.0):
                raise ValueError("alpha must be in [0, 1].")
            self.markerfacealpha = float(alpha)

        if edgecolor is not None:
            self.markeredgecolor = edgecolor

        if edgewidth is not None:
            edgewidth = float(edgewidth)
            if not math.isfinite(edgewidth):
                raise ValueError("edgewidth must be finite.")
            if edgewidth < 0:
                raise ValueError("edgewidth must be nonnegative.")
            self.markeredgewidth = edgewidth

        if edgealpha is not None:
            if not (0.0 <= edgealpha <= 1.0):
                raise ValueError("edgealpha must be in [0, 1].")
            self.markeredgealpha = float(edgealpha)

    def set_hull_options(
        self,
        color: Color | None = None,
        alpha: float | None = None,
        edgecolor: Color | None = None,
        edgewidth: float | None = None,
        edgealpha: float | None = None,
    ) -> None:
        """Set horizontal-hull display options.

        Args:
            color (Color | None, optional): Hull fill color. Defaults to None.
            alpha (float | None, optional): Hull fill alpha in [0, 1]. Defaults to None.
            edgecolor (Color | None, optional): Hull edge color. Defaults to None.
            edgewidth (float | None, optional): Hull edge width. Defaults to None.
            edgealpha (float | None, optional): Hull edge alpha in [0, 1]. Defaults to None.
        """
        if color is not None:
            self.hullcolor = color

        if alpha is not None:
            if not (0.0 <= alpha <= 1.0):
                raise ValueError("alpha must be in [0, 1].")
            self.hullalpha = float(alpha)

        if edgecolor is not None:
            self.hulledgecolor = edgecolor

        if edgewidth is not None:
            edgewidth = float(edgewidth)
            if not math.isfinite(edgewidth):
                raise ValueError("edgewidth must be finite.")
            if edgewidth < 0:
                raise ValueError("edgewidth must be nonnegative.")
            self.hulledgewidth = edgewidth

        if edgealpha is not None:
            if not (0.0 <= edgealpha <= 1.0):
                raise ValueError("edgealpha must be in [0, 1].")
            self.hulledgealpha = float(edgealpha)

    # =================
    #   DRAW HELPERS
    # =================
    def _compute_starting_ending_points_for_line_with_slope(
        self, slope: float
    ) -> tuple[float, float, float, float]:
        """Compute the start and end points for a slope-constrained guide line.

        Args:
            slope (float): Guide-line slope through the center point ``(0.5, 0.5)``.

        Returns:
            tuple[float, float, float, float]: ``(x_start, y_start, x_end, y_end)`` clipped to
                the unit square.
        """
        return line_segment_through_unit_square(slope)

    def _draw_crosshairs(self) -> None:
        """Draw crosshair guide lines centered at (0.5, 0.5)."""
        crosshair_color = self._resolved_rgba(
            self.crosshair_color,
            self.crosshair_alpha,
            field="crosshair_color",
        )

        self._ax.axvline(
            0.5,
            color=crosshair_color,
            linewidth=self.crosshair_width,
            zorder=-2,
        )
        self._ax.axhline(
            0.5,
            color=crosshair_color,
            linewidth=self.crosshair_width,
            zorder=-2,
        )

    def _draw_lines(self) -> None:
        """Draw all named and anonymous guide lines."""
        for line in self._named_lines.values():
            x_start, y_start, x_end, y_end = (
                self._compute_starting_ending_points_for_line_with_slope(line.slope)
            )
            self._ax.plot(
                [x_start, x_end],
                [y_start, y_end],
                color=self._resolved_rgba(
                    line.linecolor,
                    line.linealpha,
                    field="linecolor",
                ),
                linestyle=line.linestyle,
                linewidth=line.linewidth,
                zorder=line.zorder,
            )

        for lines in self._lines.values():
            for line in lines:
                x_start, y_start, x_end, y_end = (
                    self._compute_starting_ending_points_for_line_with_slope(line.slope)
                )
                self._ax.plot(
                    [x_start, x_end],
                    [y_start, y_end],
                    color=self._resolved_rgba(
                        line.linecolor,
                        line.linealpha,
                        field="linecolor",
                    ),
                    linestyle=line.linestyle,
                    linewidth=line.linewidth,
                    zorder=line.zorder,
                )

    def _paintball_coordinates(self) -> tuple[list[float], list[float]]:
        """Return transformed paintball coordinates in the unit square."""
        xs = [round(1.0 - v, 4) for v in self._voteshare_data]
        ys = [round(1.0 - s, 4) for s in self._seatshare_data]
        return xs, ys

    def _horizontal_hull_vertices(self) -> list[tuple[float, float]]:
        """Compute the horizontal hull vertices for the transformed paintball points."""
        xs, ys = self._paintball_coordinates()

        y_to_minmax_x: dict[float, tuple[float, float]] = {}
        for x_coord, y_coord in sorted(zip(xs, ys), key=lambda p: p[1]):
            if y_coord not in y_to_minmax_x:
                y_to_minmax_x[y_coord] = (x_coord, x_coord)
                continue

            min_x, max_x = y_to_minmax_x[y_coord]
            y_to_minmax_x[y_coord] = (min(min_x, x_coord), max(max_x, x_coord))

        sorted_y = sorted(y_to_minmax_x)
        left_side = [(y_to_minmax_x[y_val][0], y_val) for y_val in sorted_y]
        right_side = [(y_to_minmax_x[y_val][1], y_val) for y_val in reversed(sorted_y)]
        return left_side + right_side

    def _draw_points(self) -> None:
        """Draw paintball points."""
        x_coords, y_coords = self._paintball_coordinates()
        marker_facecolor = self._resolved_rgba(
            self.markerfacecolor,
            self.markerfacealpha,
            field="markerfacecolor",
        )
        marker_edgecolor = self._resolved_rgba(
            self.markeredgecolor,
            self.markeredgealpha,
            field="markeredgecolor",
        )
        self._ax.plot(
            x_coords,
            y_coords,
            linestyle="none",
            marker=self.marker,
            markersize=self.markersize,
            markerfacecolor=marker_facecolor,
            markeredgecolor=marker_edgecolor,
            markeredgewidth=self.markeredgewidth,
            zorder=2,
        )

    def _draw_horizontal_hull(self) -> None:
        """Draw the horizontal hull polygon for paintball points."""
        hull_vertices = self._horizontal_hull_vertices()

        fillcolor = self.hullcolor if self.hullcolor is not None else self.markerfacecolor
        fillalpha = self.hullalpha if self.hullalpha is not None else self.markerfacealpha
        edgecolor = self.hulledgecolor if self.hulledgecolor is not None else self.markeredgecolor
        edgealpha = self.hulledgealpha if self.hulledgealpha is not None else self.markeredgealpha
        edge_rgba = self._resolved_rgba(edgecolor, edgealpha, field="hulledgecolor")

        if len(hull_vertices) < 3:
            xs, ys = zip(*hull_vertices)
            self._ax.plot(
                xs,
                ys,
                color=edge_rgba,
                linewidth=self.hulledgewidth,
                zorder=2,
            )
            return

        x_coords = [x for x, _ in hull_vertices] + [hull_vertices[0][0]]
        y_coords = [y for _, y in hull_vertices] + [hull_vertices[0][1]]

        self._ax.fill(
            x_coords,
            y_coords,
            facecolor=self._resolved_rgba(fillcolor, fillalpha, field="hullcolor"),
            edgecolor=edge_rgba,
            linewidth=self.hulledgewidth,
            zorder=2,
        )

    def _build_plot(self) -> None:
        """Build the plot by drawing all elements in order."""
        self._draw_crosshairs()
        self._draw_lines()

        if self._draw_hull:
            self._draw_horizontal_hull()
        else:
            self._draw_points()

        self._ax.set_aspect(self.yscale / self.xscale, adjustable="box")

    @property
    def _legend_handles(self) -> list[LegendHandle]:
        """Generate legend handles for paintball data and named lines."""
        handles: list[LegendHandle] = []

        if self._draw_hull:
            fillcolor = self.hullcolor if self.hullcolor is not None else self.markerfacecolor
            fillalpha = self.hullalpha if self.hullalpha is not None else self.markerfacealpha
            edgecolor = (
                self.hulledgecolor if self.hulledgecolor is not None else self.markeredgecolor
            )
            edgealpha = (
                self.hulledgealpha if self.hulledgealpha is not None else self.markeredgealpha
            )
            handles.append(
                Patch(
                    facecolor=self._resolved_rgba(fillcolor, fillalpha, field="hullcolor"),
                    edgecolor=self._resolved_rgba(edgecolor, edgealpha, field="hulledgecolor"),
                    linewidth=self.hulledgewidth,
                    label="Horizontal Hull",
                )
            )
        else:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="none",
                    marker=self.marker,
                    markersize=self.markersize,
                    markerfacecolor=self._resolved_rgba(
                        self.markerfacecolor,
                        self.markerfacealpha,
                        field="markerfacecolor",
                    ),
                    markeredgecolor=self._resolved_rgba(
                        self.markeredgecolor,
                        self.markeredgealpha,
                        field="markeredgecolor",
                    ),
                    markeredgewidth=self.markeredgewidth,
                    label="Plan Outcomes",
                )
            )

        for line in self._named_lines.values():
            handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle=line.linestyle,
                    marker="",
                    linewidth=line.linewidth,
                    color=self._resolved_rgba(
                        line.linecolor,
                        line.linealpha,
                        field="linecolor",
                    ),
                    label=line.label,
                )
            )

        return handles

    def show(self, *, hull: bool = False, **kwargs: object) -> None:
        """Display the paintball figure.

        Args:
            hull (bool, optional): Whether to display the horizontal hull instead of points.
                Defaults to False.
            **kwargs (object): Additional keyword arguments passed to ``Figure.savefig``.
                Defaults: ``bbox_inches="tight"``, ``dpi=fig.dpi``.
        """
        previous_hull_setting = self._draw_hull
        self._draw_hull = hull
        try:
            super().show(**kwargs)
        finally:
            self._draw_hull = previous_hull_setting

    def save(self, filepath: str, *, hull: bool = False, **kwargs: object) -> None:
        """Save the paintball figure to a file.

        Args:
            filepath (str): Output image file path.
            hull (bool, optional): Whether to save the horizontal hull view.
                Defaults to False.
            **kwargs (object): Additional keyword arguments forwarded to ``Figure.savefig``.
        """
        previous_hull_setting = self._draw_hull
        self._draw_hull = hull
        try:
            super().save(filepath, **kwargs)
        finally:
            self._draw_hull = previous_hull_setting

    @property
    def hull_ax(self) -> Axes:
        """Build and return the matplotlib Axes object for the hull view."""
        previous_hull_setting = self._draw_hull
        self._draw_hull = True
        try:
            return self.ax
        finally:
            self._draw_hull = previous_hull_setting
