from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
from matplotlib.lines import Line2D

from gerrytools.logging import get_logger
from gerrytools.plotting.gerryplot import GerryPlotBase
from gerrytools.typing import Color

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class SeatsVotesData:
    """One seats-votes curve + optional overall marker.

    Attributes:
        pov_party_vote_counts (np.ndarray): An array of vote counts for the party of interest in
            each district.
        total_vote_counts (np.ndarray): An array of total vote counts in each district. Must be the
            same shape as pov_party_vote_counts.
        election_name (str): The name of the election/series, used for labeling the seats-votes
            curve in the legend.
        linecolor (str | tuple[float, float, float]): The color of the seats-votes curve.
        markercolor (str | tuple[float, float, float]): The color of the overall marker point.
        markerlabel (str): The label for the marker in the legend.
    """

    pov_party_vote_counts: np.ndarray
    total_vote_counts: np.ndarray
    election_name: str
    linecolor: str | tuple[float, float, float]
    markercolor: str | tuple[float, float, float]
    markerlabel: str

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
        linewidth (float): The width of the line.
        linestyle (str): The style of the line.
        label (str | None, optional): The label for the line in the legend. Defaults to None.
    """

    slope: float
    linecolor: Color
    linewidth: float
    linestyle: str
    label: str | None = None


class SeatsVotes(GerryPlotBase):
    """A class for creating seats-votes plots."""

    def __init__(
        self,
        figure_size: tuple[float, float] = (10, 10),
        dpi: int = 300,
        *,
        include_legend: bool = True,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
    ) -> None:
        """Initialize a SeatsVotes instance.

        Args:
            figure_size (tuple[float, float], optional): The size of the figure in inches.
                Defaults to (10, 10).
            dpi (int, optional): The dots per inch (DPI) of the figure. Defaults to 300.
            include_legend (bool, optional): Whether to include a legend in the plot.
                Defaults to True.
            xlabel (str | None, optional): The label for the x-axis. Defaults to None.
            ylabel (str | None, optional): The label for the y-axis. Defaults to None.
            title (str | None, optional): The title of the plot. Defaults to None.
        """
        super().__init__(
            figure_size=figure_size,
            dpi=dpi,
            include_legend=include_legend,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
        )

        self._sv_data_list: list[SeatsVotesData] = []
        self._line_data_list: list[SVPlotLine] = []

        self._crosshair_settings: dict[str, dict[str, Any]] | None = None
        self.update_crosshair_settings()

        self._display_election_markers = True
        self.standard_marker_color = "#daa520"
        self.standard_election_color = "#006400"

        self.linewidth = 2.5
        self.markersize = 8.0

        self._display_line_legend = True

        # Seats-votes plots are always drawn in the unit square.
        # Use GerryPlotBase deferred axis-limit setters so limits survive rebuilds.
        self.set_xlimits(0.0, 1.0)
        self.set_ylimits(0.0, 1.0)

        self.__fontsize = 16.0
        self._legend_options.fontsize = self.__fontsize
        self.set_tick_fontsize(self.__fontsize)

    def add_seat_votes_data(
        self,
        pov_party_vote_counts: Sequence[float],
        total_vote_counts: Sequence[float],
        election_name: str | None = None,
        linecolor: str | tuple[float, float, float] | None = None,
        markercolor: str | tuple[float, float, float] | None = None,
        markerlabel: str | None = None,
    ) -> None:
        """Add a seats-votes curve to the plot.

        Args:
            pov_party_vote_counts (Sequence[float]): A sequence of vote counts for the party of
                interest in each district.
            total_vote_counts (Sequence[float]): A sequence of total vote counts in each district.
                Must be the same length as pov_party_vote_counts.
            election_name (str | None, optional): The name of the election/series, used for labeling
                the seats-votes curve in the legend. Defaults to None.
            linecolor (str | tuple[float, float, float] | None, optional): The color of the
                seats-votes curve. Defaults to None, which uses
                ``self.standard_election_color``.
            markercolor (str | tuple[float, float, float] | None, optional): The color of the
                overall marker point. Defaults to None, which uses
                ``self.standard_marker_color``. Markers are shown/hidden by
                ``show_election_markers()`` / ``hide_election_markers()``.
            markerlabel (str | None, optional): The label for the election-result marker in the
                legend. If not
                provided, then all will be assigned the default marker label "Election Result"
        """

        if markercolor is None:
            self.markercolor = self.standard_marker_color

        self._sv_data_list.append(
            SeatsVotesData(
                pov_party_vote_counts=np.array(pov_party_vote_counts),
                total_vote_counts=np.array(total_vote_counts),
                election_name=(
                    election_name if election_name is not None else "Election Seats-Votes Curve"
                ),
                linecolor=linecolor if linecolor is not None else self.standard_election_color,
                markercolor=markercolor if markercolor is not None else self.standard_marker_color,
                markerlabel=markerlabel if markerlabel is not None else "Election Result",
            )
        )

    # ========================
    # ==  Cosmetic helpers  ==
    # ========================
    def update_crosshair_settings(
        self,
        x_width: float = 0.02,
        y_width: float = 0.02,
        color: str | tuple[float, float, float] = "lightgrey",
        alpha: float = 1.0,
    ) -> None:
        """Add crosshairs centered at (0.5, 0.5) to the plot.

        Args:
            x_width (float, optional): The width of the vertical crosshair line. Defaults to 0.02.
            y_width (float, optional): The width of the horizontal crosshair line. Defaults to 0.02.
            color (str | tuple[float, float, float], optional): The color of the crosshair lines.
                Defaults to "lightgrey".
            alpha (float, optional): The alpha transparency of the crosshair lines. Defaults to 1.0.
        """
        dx = x_width / 2
        dy = y_width / 2
        self._crosshair_settings = dict(
            x=dict(
                xmin=0.5 - dx,
                xmax=0.5 + dx,
                color=color,
                alpha=alpha,
                zorder=-2,
            ),
            y=dict(
                ymin=0.5 - dy,
                ymax=0.5 + dy,
                color=color,
                alpha=alpha,
                zorder=-2,
            ),
        )

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
        color: str | tuple[float, float, float] = "grey",
        linestyle: str = "--",
        linewidth: float = 2.0,
    ) -> None:
        """Add a proportionality line (y=x) to the plot.

        Args:
            color (str | tuple[float, float, float], optional): The color of the line. Defaults
                to "grey".
            linestyle (str, optional): The style of the line. Defaults to "--".
            linewidth (float, optional): The width of the line. Defaults to 2.0.
        """
        self._line_data_list.append(
            SVPlotLine(
                slope=1.0,
                linecolor=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label="Proportionality",
            )
        )

    def add_efficiency_gap_line(
        self,
        color: str | tuple[float, float, float] = "grey",
        linestyle: str = "-",
        linewidth: float = 2.0,
    ) -> None:
        """Add an Efficiency Gap line (y=2x-0.5) to the plot.

        Args:
            color (str | tuple[float, float, float], optional): The color of the line. Defaults
                to "grey".
            linestyle (str, optional): The style of the line. Defaults to "-".
            linewidth (float, optional): The width of the line. Defaults to 2.0.
        """
        self._line_data_list.append(
            SVPlotLine(
                slope=2.0,
                linecolor=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label="Efficiency Gap",
            )
        )

    def add_custom_line(
        self,
        slope: float,
        linecolor: str | tuple[float, float, float],
        linestyle: str,
        linewidth: float,
        label: str | None = None,
    ) -> None:
        """Add a custom line with the given slope to the plot.

        The line is clipped to the unit square and constrained to pass through
        the center point (0.5, 0.5).

        Args:
            slope (float): The slope of the line.
            linecolor (str | tuple[float, float, float], optional): The color of the line.
            linestyle (str, optional): The style of the line.
            linewidth (float, optional): The width of the line.
            label (str | None, optional): The label for the line in the legend. Defaults to None.
        """

        self._line_data_list.append(
            SVPlotLine(
                slope=slope,
                linecolor=linecolor,
                linestyle=linestyle,
                linewidth=linewidth,
                label=label,
            )
        )

    def _set_axis_tick_fontsize(
        self,
        *,
        axis: Literal["x", "y"],
        fontsize: float,
    ) -> None:
        """Apply fontsize while preserving any existing tick-style settings."""
        if axis == "x":
            style = self._x_tick_style
            setter = self.set_xaxis_tick_style
        else:
            style = self._y_tick_style
            setter = self.set_yaxis_tick_style

        if style is None:
            setter(size=fontsize)
            return

        setter(
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

    def set_tick_fontsize(self, fontsize: float) -> None:
        """Set the font size of the tick labels.

        Args:
            fontsize (float): The font size to set for the tick labels.
        """
        self.__fontsize = fontsize
        self._set_axis_tick_fontsize(axis="x", fontsize=fontsize)
        self._set_axis_tick_fontsize(axis="y", fontsize=fontsize)

    def set_fontsize(self, fontsize: float) -> None:
        """Set the font size of tick labels and legend text.

        Args:
            fontsize (float): Font size for ticks and legend text.
        """
        self.set_tick_fontsize(fontsize)
        self._legend_options.fontsize = fontsize

    def set_markersize(self, markersize: float) -> None:
        """Set the size of the election result markers.

        Args:
            markersize (float): The size to set for the election result markers.
        """
        self.markersize = markersize

    def set_linewidth(self, linewidth: float) -> None:
        """Set the width of the seats-votes curves.

        Args:
            linewidth (float): The width to set for the seats-votes curves.
        """
        self.linewidth = linewidth

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
        if slope == 0:
            starting_x = 0.0
            ending_x = 1.0
            starting_y = 0.5
            ending_y = 0.5
        elif slope == float("inf") or slope == float("-inf"):
            starting_x = 0.5
            ending_x = 0.5
            starting_y = 0.0
            ending_y = 1.0
        elif slope >= 1:
            starting_x = 0.5 - (0.5 / slope)
            starting_y = 0.0
            ending_x = 0.5 + (0.5 / slope)
            ending_y = 1.0
        elif 0 < slope < 1:
            starting_x = 0.0
            starting_y = 0.5 - (0.5 * slope)
            ending_x = 1.0
            ending_y = 0.5 + (0.5 * slope)
        elif -1 < slope < 0:
            starting_x = 0.0
            starting_y = 0.5 - (0.5 * slope)
            ending_x = 1.0
            ending_y = 0.5 + (0.5 * slope)
        else:
            starting_x = 0.5 - (0.5 / slope)
            starting_y = 0.0
            ending_x = 0.5 + (0.5 / slope)
            ending_y = 1.0

        starting_x = round(starting_x, 4)
        starting_y = round(starting_y, 4)
        ending_x = round(ending_x, 4)
        ending_y = round(ending_y, 4)

        return starting_x, starting_y, ending_x, ending_y

    def _draw_lines(self) -> None:
        """Draw all custom lines on the plot."""
        for line in self._line_data_list:
            x_start, y_start, x_end, y_end = (
                self._compute_starting_ending_points_for_line_with_slope(line.slope)
            )
            x_vals = [x_start, x_end]
            y_vals = [y_start, y_end]
            self._ax.plot(
                x_vals,
                y_vals,
                color=line.linecolor,
                linestyle=line.linestyle,
                linewidth=line.linewidth,
                zorder=-1,
            )

    def _draw_seats_votes_curves(self) -> None:
        """Draw the seats-votes curves on the plot."""
        for sv_series in self._sv_data_list:
            vote_shares, seat_shares = sv_series.seats_votes_curve_values()

            self._ax.step(
                vote_shares,
                seat_shares,
                where="pre",
                color=sv_series.linecolor,
                linewidth=self.linewidth,
            )

    def _draw_sv_markers(self) -> None:
        """Draw the overall election result markers on the plot."""
        for sv_series in self._sv_data_list:
            if sv_series.markercolor is not None:
                total_vote_share = float(
                    sv_series.pov_party_vote_counts.sum() / sv_series.total_vote_counts.sum()
                )
                district_vote_shares = sv_series.pov_party_vote_counts / sv_series.total_vote_counts
                total_seat_share = float(np.mean(district_vote_shares > 0.5))

                self._ax.plot(
                    total_vote_share,
                    total_seat_share,
                    marker="o",
                    linestyle="",
                    color=sv_series.markercolor,
                    markersize=self.markersize,
                )

    def _build_plot(self) -> None:
        """Build the plot by drawing all elements in the correct order."""
        self._draw_seats_votes_curves()
        self._draw_lines()
        self._ax.set_aspect("equal", adjustable="box")

        if self._display_election_markers:
            self._draw_sv_markers()

        if self._crosshair_settings is not None:
            self._ax.axvspan(**self._crosshair_settings["x"])
            self._ax.axhspan(**self._crosshair_settings["y"])

    def _get_sv_curve_legend_handles(self) -> list[Any]:
        """Generate legend handles for seats-votes curves.

        Returns:
            list[Any]: A list of legend handles for the seats-votes curves.
        """
        handles: list[Any] = []

        line_color_election_name_pairs = dict.fromkeys(
            (sdata.linecolor, sdata.election_name) for sdata in self._sv_data_list
        )
        for linecolor, election_name in line_color_election_name_pairs:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="-",
                    marker="",
                    label=election_name,
                    color=linecolor,
                )
            )

        return handles

    def _get_sv_marker_legend_handles(self) -> list[Any]:
        """Generate legend handles for election-result markers.

        Returns:
            list[Any]: A list of legend handles for election-result markers.
        """
        handles: list[Any] = []

        marker_color_marker_label_pairs = dict.fromkeys(
            (sdata.markercolor, sdata.markerlabel) for sdata in self._sv_data_list
        )
        for markercolor, markerlabel in marker_color_marker_label_pairs:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="none",
                    label=markerlabel,
                    marker="o",
                    color=markercolor,
                    markersize=self.markersize,
                )
            )

        return handles

    def _get_line_legend_handles(self) -> list[Any]:
        """Generate legend handles for custom lines.

        Returns:
            list[Any]: A list of legend handles for the custom lines.
        """
        handles: list[Any] = []

        for line in self._line_data_list:
            if line.label is not None:
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        linestyle=line.linestyle,
                        marker="",
                        label=line.label,
                        color=line.linecolor,
                    )
                )

        return handles

    @property
    def _legend_handles(self) -> list[Any]:
        """Generate legend handles for seats-votes curves, markers, and guide lines."""
        handles: list[Any] = []

        handles.extend(self._get_sv_curve_legend_handles())
        if self._display_election_markers:
            handles.extend(self._get_sv_marker_legend_handles())
        if self._display_line_legend:
            handles.extend(self._get_line_legend_handles())
        return handles
