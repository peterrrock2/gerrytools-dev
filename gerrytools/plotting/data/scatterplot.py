from dataclasses import dataclass
from typing import Sequence

import numpy as np
from matplotlib.lines import Line2D

from gerrytools.logging import get_logger
from gerrytools.plotting.data.gerryplot import GerryPlotBase
from gerrytools.plotting.mpl.marker_options import PointMarkerOptions
from gerrytools.typing import Color, LegendHandle

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class ScatterData:
    """Data for a scatterplot."""

    x: np.ndarray
    y: np.ndarray
    label: str | None
    marker_options: PointMarkerOptions

    def __post_init__(self) -> None:
        if self.x.shape != self.y.shape:
            raise ValueError("x and y must have the same shape.")
        if self.x.ndim != 1:
            raise ValueError("x and y must be 1-dimensional arrays.")
        if self.x.size == 0:
            raise ValueError("x and y must not be empty.")


class ScatterPlot(GerryPlotBase):
    """A class for creating standard scatterplots."""

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
        """Initialize a BoxPlot instance.

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
        super().__init__(
            figure_size=figure_size,
            dpi=dpi,
            include_legend=include_legend,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
        )

        self._scatter_data_list: list[ScatterData] = []
        self._labels: list[str] | None = None

    def add_scatter(
        self,
        x: Sequence[float] | None = None,
        y: Sequence[float] | None = None,
        xy_pairs: list[tuple[float, float]] | None = None,
        *,
        label: str | None = None,
        markerfacecolor: Color = "#b0b0b0",
        markerfacealpha: float | None = None,
        marker: str = "o",
        markersize: float = 6.0,
        markeredgecolor: Color | None = None,
        markeredgealpha: float | None = None,
        markeredgewidth: float = 0.0,
        zorder: int = 1,
    ) -> None:
        """Add a set of points to the scatterplot.

        Args:
            x (Sequence[float] | None): The x-coordinates of the points. Defaults to None.
            y (Sequence[float] | None): The y-coordinates of the points. Defaults to None.
            xy_pairs (list[tuple[float, float]] | None): A list of (x, y) coordinate pairs.
                If provided, x and y should be None. Defaults to None.
            label (str | None, optional): The label for the point set. Defaults to None.
            markerfacecolor (Color, optional): The face color of the markers. Defaults to "#b0b0b0"
                which is a medium gray.
            markerfacealpha (float | None, optional): The alpha value for the marker face color.
                Defaults to None.
            marker (str, optional): The marker style. Defaults to "o".
            markersize (float, optional): The size of the markers. Defaults to 6.0.
            markeredgecolor (Color | None, optional): The edge color of the markers. Defaults to
                None.
            markeredgealpha (float | None, optional): The alpha value for the marker edge color.
                Defaults to None.
            markeredgewidth (float, optional): The width of the marker edges. Defaults to 0.0.
            zorder (int, optional): The z-order of the markers. Defaults to 1.

        Raises:
            ValueError: If both xy_pairs and x/y are provided, or if neither is provided.
        """
        if xy_pairs is not None:
            if x is not None or y is not None:
                raise ValueError("Specify either xy_pairs or x and y, not both.")
            x, y = zip(*xy_pairs)

        if x is None or y is None:
            raise ValueError("Both x and y must be provided.")

        pointset_data = ScatterData(
            x=np.array(x),
            y=np.array(y),
            label=label,
            marker_options=PointMarkerOptions(
                marker=marker,
                markersize=markersize,
                markerfacecolor=markerfacecolor,
                markerfacealpha=markerfacealpha,
                markeredgecolor=markeredgecolor if markeredgecolor is not None else "none",
                markeredgealpha=markeredgealpha,
                markeredgewidth=markeredgewidth,
                zorder=zorder,
            ),
        )
        self._scatter_data_list.append(pointset_data)

    def add_point(
        self,
        x: float,
        y: float,
        *,
        label: str,
        markerfacecolor: Color = "denim",
        markerfacealpha: float | None = None,
        marker: str = "o",
        markersize: float = 6.0,
        markeredgecolor: Color | None = None,
        markeredgealpha: float | None = None,
        markeredgewidth: float = 0.0,
        zorder: int = 1,
    ) -> None:
        """Add a single point to the scatterplot.

        Args:
            x (float): The x-coordinate of the point.
            y (float): The y-coordinate of the point.
            label (str): The label for the point.
            markerfacecolor (Color, optional): The face color of the marker. Defaults to "denim".
            markerfacealpha (float | None, optional): The alpha value for the marker face color.
                Defaults to None.
            marker (str, optional): The marker style. Defaults to "o".
            markersize (float, optional): The size of the marker. Defaults to 6.0.
            markeredgecolor (Color | None, optional): The edge color of the marker. Defaults to
                None.
            markeredgealpha (float | None, optional): The alpha value for the marker edge color.
                Defaults to None.
            markeredgewidth (float, optional): The width of the marker edge. Defaults to 0.0.
            zorder (int, optional): The z-order of the marker. Defaults to 1.
        """
        self.add_scatter(
            x=[x],
            y=[y],
            label=label,
            markerfacecolor=markerfacecolor,
            markerfacealpha=markerfacealpha,
            marker=marker,
            markersize=markersize,
            markeredgecolor=markeredgecolor,
            markeredgealpha=markeredgealpha,
            markeredgewidth=markeredgewidth,
            zorder=zorder,
        )

    def _draw_points(self) -> None:
        """Draw scatterpolts on the plot axes.

        Returns:
            None
        """
        if len(self._scatter_data_list) == 0:
            return

        for sdata in self._scatter_data_list:
            self._ax.plot(
                sdata.x,
                sdata.y,
                linestyle="none",
                clip_on=True,
                rasterized=True,
                **sdata.marker_options.to_mpl_settings_dict(),
            )

    def _build_plot(self) -> None:
        """Build the scatterplot by drawing point sets."""
        self._draw_points()

    def _get_scatter_legend_handles(self) -> list[LegendHandle]:
        """Generate legend handles for point sets.

        Returns:
            list[LegendHandle]: A list of legend handles for the point sets.
        """
        handles: list[LegendHandle] = []

        for sdata in self._scatter_data_list:
            if sdata.label is None:
                continue
            handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="none",
                    label=sdata.label,
                    **sdata.marker_options.to_mpl_settings_dict(),
                )
            )

        return handles

    @property
    def _legend_handles(self) -> list[LegendHandle]:
        """Generated legend handles for boxplot and point sets."""
        handles: list[LegendHandle] = []

        handles.extend(self._get_scatter_legend_handles())
        return handles
