import math
from dataclasses import dataclass
from typing import Literal, Sequence, TypedDict

import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from gerrytools.logging import get_logger
from gerrytools.plotting.data._geometry import line_segment_through_unit_square
from gerrytools.plotting.data.gerryplot import GerryPlotBase
from gerrytools.plotting.data.options import (
    DEFAULT_EDGE_WIDTH,
    SeatsVotesLineOptions,
    SeatsVotesMarkerOptions,
)
from gerrytools.typing import Color, LegendHandle

logger = get_logger(__name__)


class _CrosshairXSettings(TypedDict):
    xmin: float
    xmax: float
    color: tuple[float, float, float, float]
    zorder: int


class _CrosshairYSettings(TypedDict):
    ymin: float
    ymax: float
    color: tuple[float, float, float, float]
    zorder: int


class _CrosshairSettings(TypedDict):
    x: _CrosshairXSettings
    y: _CrosshairYSettings


@dataclass(slots=True, frozen=True)
class SeatsVotesData:
    """One seats-votes curve + optional overall marker.

    Attributes:
        pov_party_vote_counts (np.ndarray): An array of vote counts for the party of interest in
            each district.
        total_vote_counts (np.ndarray): An array of total vote counts in each district. Must be the
            same shape as pov_party_vote_counts.
        name (str): The name of the election/series, used for labeling the seats-votes curve in
            the legend.
        linecolor (Color): The color of the seats-votes curve.
        linealpha (float | None): Optional alpha override for the seats-votes curve.
        linestyle (str): Line style for the seats-votes curve.
        linewidth (float | None): Optional line-width override for this curve.
        zorder (int): z-order of the seats-votes curve.
        markerfacecolor (Color): The color of the overall marker point.
        markerfacealpha (float | None): Optional alpha override for marker face color.
        marker (str): Marker style for election-result marker.
        markersize (float | None): Optional marker-size override for this marker.
        markeredgecolor (Color | None): Optional marker edge color.
        markeredgealpha (float | None): Optional marker edge alpha.
        markeredgewidth (float): Marker edge width.
        markerzorder (int): z-order of the election-result marker.
        markerlabel (str): The label for the marker in the legend.
    """

    pov_party_vote_counts: np.ndarray
    total_vote_counts: np.ndarray
    name: str
    linecolor: Color
    markerfacecolor: Color
    markerlabel: str
    linealpha: float | None = None
    linestyle: str = "-"
    linewidth: float | None = None
    zorder: int = 1
    markerfacealpha: float | None = None
    marker: str = "o"
    markersize: float | None = None
    markeredgecolor: Color | None = None
    markeredgealpha: float | None = None
    markeredgewidth: float = 0.0
    markerzorder: int = 2

    def __post_init__(self) -> None:
        if self.linealpha is not None:
            linealpha = float(self.linealpha)
            if not (0.0 <= linealpha <= 1.0):
                raise ValueError("linealpha must be in [0, 1].")
            object.__setattr__(self, "linealpha", linealpha)

        if self.linewidth is not None:
            linewidth = float(self.linewidth)
            if not math.isfinite(linewidth):
                raise ValueError("linewidth must be finite.")
            if linewidth < 0.0:
                raise ValueError("linewidth must be nonnegative.")
            object.__setattr__(self, "linewidth", linewidth)

        if self.markerfacealpha is not None:
            markerfacealpha = float(self.markerfacealpha)
            if not (0.0 <= markerfacealpha <= 1.0):
                raise ValueError("markerfacealpha must be in [0, 1].")
            object.__setattr__(self, "markerfacealpha", markerfacealpha)

        if self.markersize is not None:
            markersize = float(self.markersize)
            if not math.isfinite(markersize):
                raise ValueError("markersize must be finite.")
            if markersize < 0.0:
                raise ValueError("markersize must be nonnegative.")
            object.__setattr__(self, "markersize", markersize)

        if self.markeredgealpha is not None:
            markeredgealpha = float(self.markeredgealpha)
            if not (0.0 <= markeredgealpha <= 1.0):
                raise ValueError("markeredgealpha must be in [0, 1].")
            object.__setattr__(self, "markeredgealpha", markeredgealpha)

        markeredgewidth = float(self.markeredgewidth)
        if not math.isfinite(markeredgewidth):
            raise ValueError("markeredgewidth must be finite.")
        if markeredgewidth < 0.0:
            raise ValueError("markeredgewidth must be nonnegative.")
        object.__setattr__(self, "markeredgewidth", markeredgewidth)

        object.__setattr__(self, "zorder", int(self.zorder))
        object.__setattr__(self, "markerzorder", int(self.markerzorder))

    def resolved_linewidth(self, default_linewidth: float) -> float:
        """Return per-series curve width, falling back to plot default.

        Args:
            default_linewidth (float): Plot-level default line width.

        Returns:
            float: Effective curve width for this series.
        """
        return float(default_linewidth) if self.linewidth is None else float(self.linewidth)

    def resolved_markersize(self, default_markersize: float) -> float:
        """Return per-series marker size, falling back to plot default.

        Args:
            default_markersize (float): Plot-level default marker size.

        Returns:
            float: Effective marker size for this series.
        """
        return float(default_markersize) if self.markersize is None else float(self.markersize)

    def resolved_markeredgecolor(self) -> Color:
        """Return marker edge color, defaulting to marker face color."""
        return self.markerfacecolor if self.markeredgecolor is None else self.markeredgecolor

    def resolved_markeredgealpha(self) -> float | None:
        """Return marker edge alpha, defaulting to marker face alpha."""
        return self.markerfacealpha if self.markeredgealpha is None else self.markeredgealpha

    def seats_votes_curve_values(
        self,
    ) -> tuple[list[float], list[float]]:
        """
        Compute the standard "uniform swing" seats-votes step curve positions.

        vote_share_shift_positions are the x breakpoints for the step curve.
        seat_shares_shift_positions are the y values (0..1) stepped by district rank.
        """
        if self.pov_party_vote_counts.shape != self.total_vote_counts.shape:
            raise ValueError("party_votes and total_votes must have the same shape.")

        if np.any(self.total_vote_counts <= 0):
            raise ValueError("total_votes must be positive for all districts.")

        vote_shares = self.pov_party_vote_counts / self.total_vote_counts
        wgts = self.total_vote_counts

        overall_percent = float(np.sum(vote_shares * wgts) / np.sum(wgts))

        vote_share_shift_positions = (
            [0.0] + sorted([float(overall_percent - r + 0.5) for r in vote_shares]) + [1.0]
        )

        n_seats = len(vote_shares)
        seat_shares_shift_positions = [0.0] + list(map(float, np.arange(n_seats + 1) / n_seats))

        return vote_share_shift_positions, seat_shares_shift_positions


@dataclass(frozen=True)
class SVPlotLine:
    """Dataclass for storing plot line properties.

    Attributes:
        slope (float): The slope of the line.
        linecolor (Color): The color of the line.
        linealpha (float | None): Optional alpha override for the line color.
        linewidth (float): The width of the line.
        linestyle (str): The style of the line.
        zorder (int): The z-order used to draw the line.
        label (str | None, optional): The label for the line in the legend. Defaults to None.
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

        linewidth = float(self.linewidth)
        if not math.isfinite(linewidth):
            raise ValueError("linewidth must be finite.")
        if linewidth < 0.0:
            raise ValueError("linewidth must be nonnegative.")
        object.__setattr__(self, "linewidth", linewidth)

        if self.linealpha is not None:
            linealpha = float(self.linealpha)
            if not (0.0 <= linealpha <= 1.0):
                raise ValueError("linealpha must be in [0, 1].")
            object.__setattr__(self, "linealpha", linealpha)

        object.__setattr__(self, "zorder", int(self.zorder))


class SeatsVotes(GerryPlotBase):
    """A class for creating seats-votes plots."""

    def __init__(
        self,
        figure_size: tuple[float, float] | None = None,
        dpi: int | None = None,
        *,
        ax: Axes | None = None,
        include_legend: bool = True,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
    ) -> None:
        """Initialize a SeatsVotes instance.

        Args:
            figure_size (tuple[float, float] | None, optional): The size of the
                figure in inches. Defaults to ``(10, 10)`` when ``ax`` is not provided.
            dpi (int | None, optional): The dots per inch (DPI) of the figure.
                Defaults to ``300`` when ``ax`` is not provided.
            ax (matplotlib.axes.Axes | None, optional): Render onto an existing
                matplotlib ``Axes`` instead of creating a fresh figure. Defaults to None.
            include_legend (bool, optional): Whether to include a legend in the plot.
                Defaults to True.
            xlabel (str | None, optional): The label for the x-axis. Defaults to None.
            ylabel (str | None, optional): The label for the y-axis. Defaults to None.
            title (str | None, optional): The title of the plot. Defaults to None.
        """
        # SeatsVotes prefers a square 10x10 figure; only apply this default when
        # the user hasn't otherwise specified a size or supplied their own axes.
        if figure_size is None and ax is None:
            figure_size = (10, 10)
        super().__init__(
            figure_size=figure_size,
            dpi=dpi,
            ax=ax,
            include_legend=include_legend,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
        )

        self._sv_data_list: list[SeatsVotesData] = []
        self._line_data_list: list[SVPlotLine] = []

        self._crosshair_settings: _CrosshairSettings | None = None
        self.set_crosshair_options()

        self._display_election_markers = True
        self.standard_marker_color: Color = "#daa520"
        self.standard_election_color: Color = "#006400"

        self.linewidth = 2.5
        self.markersize = 8.0

        self._display_line_legend = True

        # Seats-votes plots are always drawn in the unit square.
        # Use GerryPlotBase deferred axis-limit setters so limits survive rebuilds.
        self.set_xlim(0.0, 1.0)
        self.set_ylim(0.0, 1.0)

        self.__fontsize = 16.0
        self._legend_options.fontsize = self.__fontsize
        self._set_axis_tick_fontsize(axis="x", fontsize=self.__fontsize)
        self._set_axis_tick_fontsize(axis="y", fontsize=self.__fontsize)

    def add_seat_votes_data(
        self,
        pov_party_vote_shares: Sequence[int | float] | NDArray,
        total_vote_shares: Sequence[int | float] | NDArray | None = None,
        name: str | None = None,
        *,
        line_options: SeatsVotesLineOptions | None = None,
        marker_options: SeatsVotesMarkerOptions | None = None,
        linecolor: Color | None = None,
        linealpha: float | None = None,
        linestyle: str | None = None,
        linewidth: float | None = None,
        zorder: int | None = None,
        markerfacecolor: Color | None = None,
        markerfacealpha: float | None = None,
        marker: str | None = None,
        markersize: float | None = None,
        markeredgecolor: Color | None = None,
        markeredgealpha: float | None = None,
        markeredgewidth: float | None = None,
        markerzorder: int | None = None,
        markerlabel: str | None = None,
    ) -> None:
        """Add a seats-votes curve to the plot.

        Args:
            pov_party_vote_shares (Sequence[int | float] | NDArray): A sequence or array of vote
                counts or vote shares for the party of interest in each district. Vote shares
                should be values between 0 and 1, and, if provided, `total_vote_shares` cannot be
                provided.
            total_vote_shares (Sequence[int | float] | NDArray, optional): A sequence or array of
                total vote counts or shares in each district. If None, then `pov_party_vote_shares`
                is assumed to
                be vote shares (values between 0 and 1) and total vote share is assumed to be 1.0
                for all districts. If provided, must be the same shape as `pov_party_vote_shares`.
                Defaults to None.
            name (str | None, optional): The name of the election/series, used for labeling the
                seats-votes curve in the legend. Defaults to None.
            linecolor (Color | None, optional): The color of the seats-votes curve. Defaults to
                None, which uses ``self.standard_election_color``.
            linealpha (float | None, optional): The alpha transparency for the seats-votes curve.
                Defaults to None.
            linestyle (str, optional): The line style for the seats-votes curve. Defaults to "-".
            linewidth (float | None, optional): The line width for this seats-votes curve.
                Defaults to None, which uses ``self.linewidth``.
            zorder (int, optional): The z-order of the seats-votes curve. Defaults to 1.
            markerfacecolor (Color | None, optional): The color of the
                overall marker point. Defaults to None, which uses ``self.standard_marker_color``.
                Markers are shown/hidden by ``show_election_markers()`` /
                ``hide_election_markers()``.
            markerfacealpha (float | None, optional): The alpha transparency of the marker face.
                Defaults to None.
            marker (str, optional): The marker style for the election-result marker.
                Defaults to "o".
            markersize (float | None, optional): Marker size for this data set.
                Defaults to None, which uses ``self.markersize``.
            markeredgecolor (Color | None, optional): Marker edge color. Defaults to None, which
                uses markerfacecolor.
            markeredgealpha (float | None, optional): Marker edge alpha. Defaults to None, which
                uses markerfacealpha.
            markeredgewidth (float, optional): Marker edge width. Defaults to None (unset): when a
                visible ``markeredgecolor`` is given but this is left unset, it falls back to 0.8 so
                the edge is drawn. Pass ``markeredgewidth=0`` explicitly to keep the edge hidden.
            markerzorder (int, optional): The z-order of the marker point. Defaults to 2.
            markerlabel (str | None, optional): The label for the election-result marker in the
                legend. If not
                provided, then all will be assigned the default marker label "Election Result"
        """

        if total_vote_shares is None:
            if any(v < 0 or v > 1 for v in pov_party_vote_shares):
                raise ValueError(
                    "If total_vote_shares is not provided, then pov_party_vote_shares must be "
                    "vote shares (values between 0 and 1)."
                )
            total_vote_shares = [1.0] * len(pov_party_vote_shares)

        # Resolve line styling.
        line_base = line_options if line_options is not None else SeatsVotesLineOptions()
        resolved_linecolor = (
            linecolor
            if linecolor is not None
            else (
                line_base.linecolor
                if line_base.linecolor is not None
                else self.standard_election_color
            )
        )
        resolved_linealpha = linealpha if linealpha is not None else line_base.linealpha
        resolved_linestyle = linestyle if linestyle is not None else line_base.linestyle
        resolved_linewidth = linewidth if linewidth is not None else line_base.linewidth
        resolved_zorder = zorder if zorder is not None else line_base.zorder

        # Resolve marker styling.
        marker_base = marker_options if marker_options is not None else SeatsVotesMarkerOptions()
        resolved_markerfacecolor = (
            markerfacecolor
            if markerfacecolor is not None
            else (
                marker_base.markerfacecolor
                if marker_base.markerfacecolor is not None
                else self.standard_marker_color
            )
        )
        resolved_markerfacealpha = (
            markerfacealpha if markerfacealpha is not None else marker_base.markerfacealpha
        )
        resolved_marker = marker if marker is not None else marker_base.marker
        resolved_markersize = markersize if markersize is not None else marker_base.markersize
        resolved_markeredgecolor = (
            markeredgecolor if markeredgecolor is not None else marker_base.markeredgecolor
        )
        resolved_markeredgealpha = (
            markeredgealpha if markeredgealpha is not None else marker_base.markeredgealpha
        )
        resolved_markeredgewidth = (
            markeredgewidth if markeredgewidth is not None else marker_base.markeredgewidth
        )
        resolved_markerzorder = (
            markerzorder if markerzorder is not None else marker_base.markerzorder
        )

        # A visible marker edge color with zero width draws nothing. If the caller named an edge
        # color but left ``markeredgewidth`` unset, fall back to a default so the edge shows. An
        # explicit ``markeredgewidth=0`` is left untouched and still hides the edge.
        if (
            markeredgewidth is None
            and resolved_markeredgewidth == 0.0
            and resolved_markeredgecolor is not None
            and str(resolved_markeredgecolor).strip().lower() != "none"
        ):
            resolved_markeredgewidth = DEFAULT_EDGE_WIDTH

        self._sv_data_list.append(
            SeatsVotesData(
                pov_party_vote_counts=np.array(pov_party_vote_shares),
                total_vote_counts=np.array(total_vote_shares),
                name=name if name is not None else "Election Seats-Votes Curve",
                linecolor=resolved_linecolor,
                linealpha=resolved_linealpha,
                linestyle=resolved_linestyle,
                linewidth=resolved_linewidth,
                zorder=resolved_zorder,
                markerfacecolor=resolved_markerfacecolor,
                markerfacealpha=resolved_markerfacealpha,
                marker=resolved_marker,
                markersize=resolved_markersize,
                markeredgecolor=resolved_markeredgecolor,
                markeredgealpha=resolved_markeredgealpha,
                markeredgewidth=resolved_markeredgewidth,
                markerzorder=resolved_markerzorder,
                markerlabel=markerlabel if markerlabel is not None else "Election Result",
            )
        )

    # ========================
    # ==  Cosmetic helpers  ==
    # ========================
    def set_crosshair_options(
        self,
        *,
        x_width: float = 0.02,
        y_width: float = 0.02,
        color: Color = "lightgrey",
        alpha: float = 1.0,
    ) -> None:
        """Add crosshairs centered at (0.5, 0.5) to the plot.

        Args:
            x_width (float, optional): The width of the vertical crosshair line. Defaults to 0.02.
            y_width (float, optional): The width of the horizontal crosshair line. Defaults to 0.02.
            color (Color, optional): The color of the crosshair lines.
                Defaults to "lightgrey".
            alpha (float, optional): The alpha transparency of the crosshair lines. Defaults to 1.0.
        """
        dx = x_width / 2
        dy = y_width / 2
        crosshair_settings: _CrosshairSettings = {
            "x": {
                "xmin": 0.5 - dx,
                "xmax": 0.5 + dx,
                "color": self._resolved_rgba(color=color, field="crosshair_color", alpha=alpha),
                "zorder": -2,
            },
            "y": {
                "ymin": 0.5 - dy,
                "ymax": 0.5 + dy,
                "color": self._resolved_rgba(color=color, field="crosshair_color", alpha=alpha),
                "zorder": -2,
            },
        }
        self._crosshair_settings = crosshair_settings

    def remove_crosshairs(self) -> None:
        """Remove crosshairs from the plot."""
        self._crosshair_settings = None

    def show_election_markers(self) -> None:
        """Whether to show the overall election result markers."""
        self._display_election_markers = True

    def hide_election_markers(self) -> None:
        """Whether to hide the overall election result markers."""
        self._display_election_markers = False

    def show_additional_lines_in_legend(self) -> None:
        """Whether to include lines in the legend."""
        self._display_line_legend = True

    def hide_additional_lines_in_legend(self) -> None:
        """Whether to hide lines from the legend."""
        self._display_line_legend = False

    def add_proportionality_line(
        self,
        *,
        color: Color = "grey",
        linealpha: float | None = None,
        linestyle: str = "--",
        linewidth: float = 2.0,
        zorder: int = -1,
        name: str | None = None,
    ) -> None:
        """Add a proportionality line (y=x) to the plot.

        Args:
            color (Color, optional): The color of the line. Defaults to "grey".
            linealpha (float | None, optional): The alpha transparency of the line.
                Defaults to None.
            linestyle (str, optional): The style of the line. Defaults to "--".
            linewidth (float, optional): The width of the line. Defaults to 2.0.
            zorder (int, optional): The z-order of the line. Defaults to -1.
            name (str | None, optional): The legend label for the line. Defaults to
                "Proportionality".
        """
        self._line_data_list.append(
            SVPlotLine(
                slope=1.0,
                linecolor=color,
                linealpha=linealpha,
                linestyle=linestyle,
                linewidth=linewidth,
                zorder=zorder,
                label=name if name is not None else "Proportionality",
            )
        )

    def add_efficiency_gap_line(
        self,
        *,
        color: Color = "grey",
        linealpha: float | None = None,
        linestyle: str = "-",
        linewidth: float = 2.0,
        zorder: int = -1,
        name: str | None = None,
    ) -> None:
        """Add an Efficiency Gap line (y=2x-0.5) to the plot.

        Args:
            color (Color, optional): The color of the line. Defaults to "grey".
            linealpha (float | None, optional): The alpha transparency of the line.
                Defaults to None.
            linestyle (str, optional): The style of the line. Defaults to "-".
            linewidth (float, optional): The width of the line. Defaults to 2.0.
            zorder (int, optional): The z-order of the line. Defaults to -1.
            name (str | None, optional): The legend label for the line. Defaults to
                "Efficiency Gap".
        """
        self._line_data_list.append(
            SVPlotLine(
                slope=2.0,
                linecolor=color,
                linealpha=linealpha,
                linestyle=linestyle,
                linewidth=linewidth,
                zorder=zorder,
                label=name if name is not None else "Efficiency Gap",
            )
        )

    def add_custom_line(
        self,
        slope: float,
        *,
        linecolor: Color,
        linealpha: float | None = None,
        linestyle: str,
        linewidth: float,
        zorder: int = -1,
        label: str | None = None,
        name: str | None = None,
    ) -> None:
        """Add a custom line with the given slope to the plot.

        The line is clipped to the unit square and constrained to pass through
        the center point (0.5, 0.5).

        Args:
            slope (float): The slope of the line.
            linecolor (Color): The color of the line.
            linealpha (float | None, optional): The alpha transparency of the line.
                Defaults to None.
            linestyle (str): The style of the line.
            linewidth (float): The width of the line.
            zorder (int, optional): The z-order of the line. Defaults to -1.
            label (str | None, optional): The label for the line in the legend. Defaults to None.
            name (str | None, optional): Alias for ``label`` for consistency with other plotting
                setter APIs. Defaults to None.
        """
        if label is not None and name is not None and label != name:
            raise ValueError("name and label must match if both are provided.")
        legend_label = name if name is not None else label

        self._line_data_list.append(
            SVPlotLine(
                slope=slope,
                linecolor=linecolor,
                linealpha=linealpha,
                linestyle=linestyle,
                linewidth=linewidth,
                zorder=zorder,
                label=legend_label,
            )
        )

    def _set_axis_tick_fontsize(
        self,
        *,
        axis: Literal["x", "y"],
        fontsize: float,
    ) -> None:
        """Apply tick font size while preserving other configured tick-style fields.

        Args:
            axis (Literal["x", "y"]): Axis to update.
            fontsize (float): New tick font size in points.

        Returns:
            None
        """
        if axis == "x":
            style = self._x_tick_style
            style_setter = self.set_xaxis_tick_style
        else:
            style = self._y_tick_style
            style_setter = self.set_yaxis_tick_style

        if style is None:
            style_setter(size=fontsize)
            return

        style_setter(
            size=fontsize,
            rotation=style.rotation,
            fontcolor=style.fontcolor,
            fontalpha=style.fontalpha,
            tickcolor=style.tickcolor,
            tickalpha=style.tickalpha,
            fontweight=style.fontweight,
            fontstyle=style.fontstyle,
            fontfamily=style.fontfamily,
            ticktype=style.ticktype,
        )

    def _compute_starting_ending_points_for_line_with_slope(
        self, slope: float
    ) -> tuple[float, float, float, float]:
        """Compute the starting and ending points for a line with the given slope.

        The line is drawn within the unit square from (0,0) to (1,1) and must pass through
        the center point (0.5, 0.5).

        Args:
            slope (float): The slope of the line.

        Returns:
            tuple[float, float, float, float]: The starting and ending points of the line
                in the format (starting_x, starting_y, ending_x, ending_y).
        """
        return line_segment_through_unit_square(slope)

    def _draw_lines(self) -> None:
        """Draw all custom lines on the plot."""
        for line in self._line_data_list:
            x_start, y_start, x_end, y_end = (
                self._compute_starting_ending_points_for_line_with_slope(line.slope)
            )
            x_vals = [x_start, x_end]
            y_vals = [y_start, y_end]
            line_artists = self._ax.plot(
                x_vals,
                y_vals,
                color=self._resolved_rgba(
                    line.linecolor,
                    alpha=line.linealpha,
                    field="linecolor",
                ),
                linestyle=line.linestyle,
                linewidth=line.linewidth,
                zorder=line.zorder,
            )
            self._artists.track(line_artists)

    def _draw_seats_votes_curves(self) -> None:
        """Draw the seats-votes curves on the plot."""
        for sv_series in self._sv_data_list:
            vote_shares, seat_shares = sv_series.seats_votes_curve_values()

            # ax.step returns a list[Line2D] just like ax.plot.
            step_artists = self._ax.step(
                vote_shares,
                seat_shares,
                where="pre",
                color=self._resolved_rgba(
                    sv_series.linecolor,
                    alpha=sv_series.linealpha,
                    field="linecolor",
                ),
                linestyle=sv_series.linestyle,
                linewidth=sv_series.resolved_linewidth(self.linewidth),
                zorder=sv_series.zorder,
            )
            self._artists.track(step_artists)

    def _draw_sv_markers(self) -> None:
        """Draw the overall election result markers on the plot."""
        for sv_series in self._sv_data_list:
            total_vote_share = float(
                sv_series.pov_party_vote_counts.sum() / sv_series.total_vote_counts.sum()
            )
            district_vote_shares = sv_series.pov_party_vote_counts / sv_series.total_vote_counts
            total_seat_share = float(np.mean(district_vote_shares > 0.5))

            marker_artists = self._ax.plot(
                total_vote_share,
                total_seat_share,
                marker=sv_series.marker,
                linestyle="",
                markerfacecolor=self._resolved_rgba(
                    sv_series.markerfacecolor,
                    alpha=sv_series.markerfacealpha,
                    field="markerfacecolor",
                ),
                markeredgecolor=self._resolved_rgba(
                    sv_series.resolved_markeredgecolor(),
                    alpha=sv_series.resolved_markeredgealpha(),
                    field="markeredgecolor",
                ),
                markeredgewidth=sv_series.markeredgewidth,
                markersize=sv_series.resolved_markersize(self.markersize),
                zorder=sv_series.markerzorder,
            )
            self._artists.track(marker_artists)

    def _apply_aspect_now(self) -> None:
        """SeatsVotes uses a 1:1 aspect ratio so the unit-square plotting region
        is geometrically faithful to the seats-vs-votes interpretation.
        """
        self._ax.set_aspect("equal", adjustable="box")

    def _build_plot(self) -> None:
        """Build the plot by drawing all elements in the correct order."""
        self._draw_seats_votes_curves()
        self._draw_lines()

        if self._display_election_markers:
            self._draw_sv_markers()

        if self._crosshair_settings is not None:
            x_settings = self._crosshair_settings["x"]
            y_settings = self._crosshair_settings["y"]
            vspan = self._ax.axvspan(
                xmin=x_settings["xmin"],
                xmax=x_settings["xmax"],
                color=x_settings["color"],
                zorder=x_settings["zorder"],
            )
            self._artists.track(vspan)
            hspan = self._ax.axhspan(
                ymin=y_settings["ymin"],
                ymax=y_settings["ymax"],
                color=y_settings["color"],
                zorder=y_settings["zorder"],
            )
            self._artists.track(hspan)

    def _get_sv_curve_legend_handles(self) -> list[LegendHandle]:
        """Generate legend handles for seats-votes curves.

        Returns:
            list[LegendHandle]: A list of legend handles for the seats-votes curves.
        """
        handles: list[LegendHandle] = []

        line_style_name_tuples = dict.fromkeys(
            (
                sdata.linecolor,
                sdata.linealpha,
                sdata.linestyle,
                sdata.resolved_linewidth(self.linewidth),
                sdata.name,
            )
            for sdata in self._sv_data_list
        )
        for linecolor, linealpha, linestyle, linewidth, name in line_style_name_tuples:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle=linestyle,
                    marker="",
                    label=name,
                    color=self._resolved_rgba(
                        linecolor,
                        alpha=linealpha,
                        field="linecolor",
                    ),
                    linewidth=linewidth,
                )
            )

        return handles

    def _get_sv_marker_legend_handles(self) -> list[LegendHandle]:
        """Generate legend handles for election-result markers.

        Returns:
            list[LegendHandle]: A list of legend handles for election-result markers.
        """
        handles: list[LegendHandle] = []

        marker_style_label_tuples = dict.fromkeys(
            (
                sdata.markerfacecolor,
                sdata.markerfacealpha,
                sdata.marker,
                sdata.resolved_markersize(self.markersize),
                sdata.resolved_markeredgecolor(),
                sdata.resolved_markeredgealpha(),
                sdata.markeredgewidth,
                sdata.markerlabel,
            )
            for sdata in self._sv_data_list
        )
        for (
            markerfacecolor,
            markerfacealpha,
            marker,
            markersize,
            markeredgecolor,
            markeredgealpha,
            markeredgewidth,
            markerlabel,
        ) in marker_style_label_tuples:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="none",
                    label=markerlabel,
                    marker=marker,
                    markerfacecolor=self._resolved_rgba(
                        markerfacecolor,
                        alpha=markerfacealpha,
                        field="markerfacecolor",
                    ),
                    markeredgecolor=self._resolved_rgba(
                        markeredgecolor,
                        alpha=markeredgealpha,
                        field="markeredgecolor",
                    ),
                    markeredgewidth=markeredgewidth,
                    markersize=markersize,
                )
            )

        return handles

    def _get_line_legend_handles(self) -> list[LegendHandle]:
        """Generate legend handles for custom lines.

        Returns:
            list[LegendHandle]: A list of legend handles for the custom lines.
        """
        handles: list[LegendHandle] = []

        for line in self._line_data_list:
            if line.label is not None:
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        linestyle=line.linestyle,
                        marker="",
                        label=line.label,
                        color=self._resolved_rgba(
                            line.linecolor,
                            alpha=line.linealpha,
                            field="linecolor",
                        ),
                        linewidth=line.linewidth,
                    )
                )

        return handles

    @property
    def _legend_handles(self) -> list[LegendHandle]:
        """Generate legend handles for seats-votes curves, markers, and guide lines."""
        handles: list[LegendHandle] = []

        handles.extend(self._get_sv_curve_legend_handles())
        if self._display_election_markers:
            handles.extend(self._get_sv_marker_legend_handles())
        if self._display_line_legend:
            handles.extend(self._get_line_legend_handles())
        return handles
