from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import chain
from typing import Any, Iterable
from warnings import warn

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Patch
from numpy.typing import NDArray

from gerrytools.colors import resolve_color_and_alpha
from gerrytools.logging import get_logger
from gerrytools.plotting.gerryplot import GerryPlotBase, PointMarkerSettings
from gerrytools.typing import BinsType, Color, HistType

logger = get_logger(__name__)


def _coerce_to_1d_float_array(
    values: Any, *, column: str | None = None, field: str
) -> NDArray[np.float64]:
    """Coerce various inputs into a 1D float ndarray (no finite-filtering).

    Args:
        values: Input values; can be an iterable, numpy array, pandas Series, or pandas

    Kwargs:
        column: If ``values`` is a DataFrame, the column name to extract.
        field: Name of the field for error messages.

    Returns:
        1D ndarray of float values.
    """
    if values is None:
        raise ValueError(f"{field}: cannot be None.")

    if isinstance(values, pd.DataFrame):
        if column is None:
            if values.shape[1] != 1:
                raise ValueError(
                    f"{field}: DataFrame input must have exactly one column or pass column=..."
                )
            ser = values.iloc[:, 0]
        else:
            if column not in values.columns:
                raise ValueError(f"{field}: column {column!r} not found in DataFrame.")
            ser = values[column]
        arr = ser.to_numpy(dtype=float)
    elif isinstance(values, pd.Series):
        arr = values.to_numpy(dtype=float)
    elif np.isscalar(values):
        arr = np.array([float(values)], dtype=float)
    elif isinstance(values, np.ndarray):
        arr = np.asarray(values, dtype=float)
    elif isinstance(values, (list, tuple)):
        arr = np.asarray(values, dtype=float)
    else:
        # generators/iterators need materializing for numpy coercion
        arr = np.asarray(list(values), dtype=float)

    arr = np.asarray(arr, dtype=float).ravel()
    return arr


def _coerce_values_and_weights(
    values: Any,
    *,
    weights: Any | None,
    column: str | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Coerce values and weights together, applying the SAME finite-mask
    to preserve alignment.
    """
    vals_raw = _coerce_to_1d_float_array(values, column=column, field="values")
    if vals_raw.size == 0:
        raise ValueError("values: must have at least one entry.")

    mask = np.isfinite(vals_raw)
    vals = vals_raw[mask]
    if vals.size == 0:
        raise ValueError("values: must have at least one finite entry.")

    if weights is None:
        wts = np.ones(vals.shape[0], dtype=float)
    else:
        w_raw = _coerce_to_1d_float_array(weights, column=None, field="weights")
        if w_raw.size != vals_raw.size:
            raise ValueError("weights must have the same length as values (before filtering).")

        wts = w_raw[mask]

        if not np.all(np.isfinite(wts)):
            raise ValueError("weights must be finite wherever values are finite.")

    return vals, wts


def _coerce_to_1d_finite_float_array(
    values: Any, *, column: str | None = None, field: str
) -> NDArray[np.float64]:
    """Coerce various inputs into a finite 1D float ndarray.

    Args:
        values: Input values; can be an iterable, numpy array, pandas Series, or pandas

    Kwargs:
        column: If ``values`` is a DataFrame, the column name to extract.
        field: Name of the field for error messages.

    Returns:
        1D ndarray of finite float values.

    Raises:
        ValueError: If input cannot be coerced to 1D float array.
    """
    arr = _coerce_to_1d_float_array(values, column=column, field=field)
    return arr[np.isfinite(arr)]


@dataclass(frozen=True)
class HistPointList:
    """A dataclass representing a list of points to be plotted on a histogram figure.

    Attributes:
        name (str): The name of the point set.
        values_list (list[float]): A list of float values representing the points to be plotted.
        point_data (PointMarkerSettings): Settings for how the points should be marked on the plot
    """

    name: str
    values: NDArray[np.float64]
    point_data: PointMarkerSettings
    y_offset: float = 0.02
    centered: bool = False


@dataclass(frozen=True)
class HistogramData:
    """One histogram layer/series.

    Attributes:
        name: Name of the histogram series.
        values: 1D array of values to histogram.
        weights: Optional 1D array of weights for the values.
        facecolor: Fill color for the histogram bars.
        facealpha: Alpha transparency for the fill color.
        edgecolor: Edge color for the histogram bars.
        edgealpha: Alpha transparency for the edge color.
        linewidth: Line width for the histogram bar edges.
        zorder: Z-order for layering the histogram in the plot.
    """

    name: str
    values: NDArray[np.float64]
    weights: NDArray[np.float64]

    facecolor: Color = "ensemble:recom"
    facealpha: float | None = None

    edgecolor: Color = "black"
    edgealpha: float | None = None
    linewidth: float = 0.8

    zorder: int = 0

    def __post_init__(self) -> None:
        vals = np.asarray(self.values, dtype=float).ravel()

        if not np.all(np.isfinite(vals)):
            raise ValueError(f"HistogramData {self.name!r}: values must be finite.")

        if vals.size == 0:
            raise ValueError(f"HistogramData {self.name!r}: values has no entries.")
        object.__setattr__(self, "values", vals)

        w_arr = np.asarray(self.weights, dtype=float).ravel()
        if w_arr.size != vals.size:
            raise ValueError("weights must have the same length as values.")
        if not np.all(np.isfinite(w_arr)):
            raise ValueError(f"HistogramData {self.name!r}: weights must be finite.")
        object.__setattr__(self, "weights", w_arr)

        lw = float(self.linewidth)
        if not math.isfinite(lw):
            raise ValueError("linewidth must be finite")
        if lw < 0:
            raise ValueError("linewidth must be nonnegative")
        object.__setattr__(self, "linewidth", lw)

        resolved_fc, resolved_a = resolve_color_and_alpha(
            self.facecolor,
            alpha=self.facealpha,
            allow_none=True,
            field="facecolor",
            owner=f"HistogramData {self.name}",
            logger=logger,
        )
        object.__setattr__(self, "facecolor", resolved_fc)
        object.__setattr__(self, "facealpha", resolved_a)

        resolved_ec, resolved_ea = resolve_color_and_alpha(
            self.edgecolor,
            alpha=self.edgealpha,
            allow_none=True,
            field="edgecolor",
            owner=f"HistogramData {self.name}",
            logger=logger,
        )
        object.__setattr__(self, "edgecolor", resolved_ec)
        object.__setattr__(self, "edgealpha", resolved_ea)

        if resolved_ec.lower() == "none" and lw > 0:
            logger.debug(
                "For HistogramData %s: edge_color is 'none' but linewidth is %s>0; setting linewidth to 0.",
                self.name,
                lw,
            )
            object.__setattr__(self, "linewidth", 0.0)

        object.__setattr__(self, "zorder", int(self.zorder))


class Histogram(GerryPlotBase):
    """Overlayed histogram comparison figure.

    Typical usage:
        h = Histogram()
        h.add_histogram(df["ensemble1"], name="Ensemble", facecolor="denim", facealpha=0.35)
        h.add_histogram(df["ensemble2"], name="Plan", histtype="outline", facecolor="black", facealpha=1.0)
        h.add_vertical_lines(plan_value, linecolor="black", name="Plan value")
        h.show()
    """

    def __init__(
        self,
        figure_size: tuple[float, float] = (10, 6),
        dpi: int = 300,
        *,
        include_legend: bool = True,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        grid: bool = False,
        hide_warnings: bool = False,
    ) -> None:
        super().__init__(figure_size=figure_size, dpi=dpi, include_legend=include_legend)
        self.hide_warnings = hide_warnings

        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.grid = grid
        self._bins: BinsType | None = None
        self._bin_alignment = "edge"
        self.as_denisty_plot = False
        self._binwidth: float | None = None

        self._hist_data_dict: dict[str, list[HistogramData]] = {
            "overlay": [],
            "weave": [],
            "outline": [],
            "stack": [],
        }
        self._histpointlist_list: list[HistPointList] = []

    def center_data_on_bin_edges(self) -> None:
        """Center histogram data on bin edges."""
        self._bin_alignment = "center"

    def set_bins(self, bins: BinsType | None) -> None:
        """Set the bins for the histogram.

        Bins can be specified either as an array of bin edges, an integer number of bins,
        or a string (e.g., 'auto') to use numpy's histogram binning strategies. The binning
        strategies possible strategies are as follows:

        - ‘auto’: Minimum bin width between the ‘sturges’ and ‘fd’ estimators. Provides good
            all-around performance.
        - ‘fd’ (Freedman Diaconis Estimator): Robust (resilient to outliers) estimator that
            takes into account data variability and data size.
        - ‘doane’: An improved version of Sturges’ estimator that works better with non-normal
            datasets.
        - ‘scott’: Less robust estimator that takes into account data variability and data size.
        - ‘stone’: Estimator based on leave-one-out cross-validation estimate of the integrated
            squared error. Can be regarded as a generalization of Scott’s rule.
        - ‘rice’: Estimator does not take variability into account, only data size. Commonly
            overestimates number of bins required.
        - ‘sturges’: R’s default method, only accounts for data size. Only optimal for gaussian
            data and underestimates number of bins for large non-gaussian datasets.
        - ‘sqrt’: Square root (of data size) estimator, used by Excel and other programs for its
            speed and simplicity.

        Args:
            bins (BinsType): Bin specification (array of bin edges, integer number of bins, or
                string specifying binning strategy). If None, defaults to 'auto'.

        Returns:
            None
        """
        self._bins = bins

    def set_bins_by_width(self, binwidth: float | None) -> None:
        """Set histogram bins by specifying a fixed bin width.

        The histogram bins will be computed automatically based on the minimum and maximum
        values across all histogram sets added to the figure.

        Args:
            binwidth (float | None): Desired width of each histogram bin. If None, the
                histogram bins will be computed automatically using numpy's default binning
                strategy ('auto').
        """
        self._bins = None
        self._binwidth = binwidth

    def clear_histograms(self) -> None:
        """Clear all histogram sets."""
        for key in self._hist_data_dict:
            self._hist_data_dict[key].clear()

    def transform_to_density(self) -> None:
        """Transform the histogram to a density plot."""
        self.as_denisty_plot = True

    def add_histogram(
        self,
        values: Iterable[float] | NDArray | pd.Series | pd.DataFrame,
        *,
        weights: Iterable[float] | NDArray | None = None,
        column: str | None = None,
        name: str | None = None,
        facecolor: Color = "denim",
        facealpha: float | None = None,
        edgecolor: Color = "none",
        edgealpha: float | None = None,
        linewidth: float = 0.0,
        histtype: HistType = "overlay",
        zorder: int = 2,
    ) -> None:
        """Add a histogram to the figure.

        Note that multiple histograms can be added to the figure, and they will be
        rendered according to the specified ``histtype``.

        - 'overlay': Histograms are drawn on top of each other, with no stacking.
        - 'stack': Histograms are stacked on top of each other.
        - 'weave': Histograms are drawn side-by-side, with each histogram bar
            divided into equal-width segments for each histogram.
        - 'outline': Histograms are drawn as outlines only, with no fill. Effectively
            a skyline plot.

        Note:
            Each type of histogram ('overlay', 'stack', 'weave', 'outline') maintains its own
            separate list of histogram data. When adding a histogram, it is added to the list
            corresponding to the specified ``histtype``. Therefore, a histogram of type 'weave'
            is added to a plot containing both 'overlay' histograms and 'weave' histograms,
            only the 'weave' histograms will be drawn in the 'weave' style; the 'overlay'
            histograms will be drawn in the 'overlay' style and will not interact with the 'weave'
            histograms. Similarly, 'stack' histograms are stacked only with other 'stack'
            histograms, and the stacking order is determined by the order in which they were added.

        Note:
            When using 'outline' histograms, it is recommended to set ``linewidth`` to
            a positive value (e.g., 0.8) to ensure the outline is visible. Additionally,
            setting ``facecolor`` to 'none' and ``edgecolor`` to a visible color (e.g., 'black')
            is advisable for clarity.


        Args:
            values: Iterable of float values, numpy array, pandas Series, or pandas DataFrame
                containing the data to be turned into a histogram.

        Kwargs:
            weights (Iterable[float] | NDArray | None, optional): Optional weights for
                the histogram values. Defaults to None.
            column (str | None, optional): The column name to use if values is a DataFrame.
            name (str | None, optional): The name of the point set. Defaults to None.
            facecolor (Color, optional): The face color of the points. Defaults to "black".
            facealpha (float | None, optional): The alpha transparency of the points.
                Defaults to None.
            edgecolor (Color, optional): The edge color of the histogram bars. Defaults to "black".
            edgealpha (float | None, optional): The alpha transparency of the histogram bar edges.
                Defaults to None.
            linewidth (float, optional): The width of the histogram bar edges. Defaults to 0.0.
            histtype (HistType, optional): The type of histogram to add. Must be one of
                'overlay', 'stack', 'weave', 'outline'. Defaults to 'overlay'.
            zorder (int, optional): The z-order of the points. Defaults to 2.
        """
        vals, wts = _coerce_values_and_weights(values, weights=weights, column=column)

        hist_list = self._hist_data_dict.get(histtype, None)
        if hist_list is None:
            raise ValueError(
                f"Invalid histtype {histtype!r}; must be one of"
                "'overlay', 'stack', 'weave', 'outline'."
            )

        if histtype == "outline":
            if linewidth <= 0.0:
                if not self.hide_warnings:
                    warn(
                        "Outline histogram specified with linewidth <= 0; setting linewidth "
                        "to 0.8.",
                        UserWarning,
                    )
                linewidth = 0.8
            if facecolor != "none":
                if not self.hide_warnings:
                    warn(
                        "Outline histogram specified with facecolor != 'none'; setting "
                        "facecolor to 'none'.",
                        UserWarning,
                    )
                facecolor = "none"

            if edgecolor == "none":
                if not self.hide_warnings:
                    warn(
                        "Outline histogram specified with edgecolor 'none'; setting "
                        "edgecolor to 'black'.",
                        UserWarning,
                    )
                edgecolor = "black"

        set_name = name or f"{histtype.capitalize()} histogram {len(hist_list) + 1}"
        hist_list.append(
            HistogramData(
                name=set_name,
                values=vals,
                weights=wts,
                facecolor=facecolor,
                facealpha=facealpha,
                edgecolor=edgecolor,
                edgealpha=edgealpha,
                linewidth=linewidth,
                zorder=zorder,
            )
        )

    def add_points_above(
        self,
        values: float | list[float] | pd.Series | pd.DataFrame | NDArray,
        *,
        column: str | None = None,
        name: str | None = None,
        facecolor: Color = "black",
        facealpha: float | None = None,
        marker: str = "o",
        markersize: float = 7.0,
        markeredgecolor: Color = "black",
        markeredgealpha: float | None = None,
        markeredgewidth: float = 0.8,
        zorder: int = 3,
        y_offset: float = 0.0,
        centered_on_bin: bool = False,
    ) -> None:
        """Add a set of points to the figure.

        Args:
            values (dict[str, float] | list[float] | pd.Series | pd.DataFrame):
                The pointset values. Can be a dictionary mapping labels to values,
                a list of values, a Series, or a DataFrame.

        Kwargs:
            column (str | None, optional): The column name to use if values is a DataFrame.
            name (str | None, optional): The name of the point set. Defaults to None.
            facecolor (Color, optional): The face color of the points. Defaults to "black".
            facealpha (float | None, optional): The alpha transparency of the points.
                Defaults to None.
            marker (str, optional): The marker style for the points. Defaults to "o".
            markersize (float, optional): The size of the point markers. Defaults to 7.0.
            markeredgecolor (Color, optional): The edge color of the point markers.
                Defaults to "black".
            markeredgealpha (float | None, optional): The alpha transparency of the pointset
                marker edges. Defaults to None.
            markeredgewidth (float, optional): The width of the point marker edges.
                Defaults to 0.8.
            zorder (int, optional): The z-order of the points. Defaults to 2.
            y_offset (float | None, optional): An absolute x-offset from category center.
                Defaults to None.
            centered_on_bin (bool, optional): If True, center the points on the histogram
                bins. Defaults to False.

        Returns:
            None
        """
        vals = _coerce_to_1d_finite_float_array(values, column=column, field="values")
        marker_name = name or f"Point Marker {len(self._histpointlist_list) + 1}"
        self._histpointlist_list.append(
            HistPointList(
                name=marker_name,
                values=vals,
                point_data=PointMarkerSettings(
                    markerfacecolor=facecolor,
                    markerfacealpha=facealpha,
                    marker=marker,
                    markersize=markersize,
                    markeredgecolor=markeredgecolor,
                    markeredgealpha=markeredgealpha,
                    markeredgewidth=markeredgewidth,
                    zorder=zorder,
                ),
                y_offset=y_offset,
                centered=centered_on_bin,
            )
        )

    def _compute_bins(self) -> NDArray[np.float64]:
        """Compute histogram bins based on current settings and data."""
        bins = self._bins
        if bins is None and self._binwidth is None:
            bins = "auto"

        all_values = np.concatenate(
            [hdata.values for hdata in chain(*self._hist_data_dict.values())]
        )
        if bins is None and self._binwidth is not None:
            minscore, maxscore = np.min(all_values), np.max(all_values)
            binwidth = self._binwidth
            bins = np.arange(minscore, maxscore + 2 * binwidth, binwidth)

        if bins is None:
            raise RuntimeError("Failed to compute histogram bins.")

        bins = np.histogram_bin_edges(all_values, bins=bins)
        return bins

    def _draw_histograms(self) -> None:
        """Draw the histograms on the plot."""
        bin_edges = self._compute_bins()
        bin_widths = np.diff(bin_edges)

        if self._bin_alignment == "center" and not np.allclose(bin_widths, bin_widths[0]):
            raise ValueError(
                "Cannot center histogram data on bin edges when bin widths are not uniform."
            )

        for histtype, histlist in self._hist_data_dict.items():
            hist_bottoms = np.zeros(len(bin_edges) - 1)
            n_bins_per_bar = 1
            if histtype in ("weave", "stack"):
                for hdata in histlist:
                    if hdata.linewidth > 0.0 and not self.hide_warnings:
                        warn(
                            f"{histtype.capitalize()} histogram {hdata.name!r} has linewidth > 0; "
                            "line edges will overlap in the plot.",
                            UserWarning,
                        )

            if histtype == "weave":
                n_bins_per_bar = len(histlist)

            for i, hdata in enumerate(histlist):
                hist_heights = np.histogram(
                    hdata.values,
                    bins=bin_edges,
                    weights=hdata.weights,
                    density=self.as_denisty_plot,
                )[0]

                # Special case for outline histograms that does the outline only (no internal
                # vertical lines)
                if histtype == "outline":
                    offset = 0.0
                    if self._bin_alignment == "center":
                        offset = 0.5 * bin_widths[0]
                    self._ax.stairs(
                        hist_heights,
                        bin_edges - offset,
                        fill=False,
                        linewidth=hdata.linewidth,
                        edgecolor=(
                            "none"
                            if hdata.edgecolor == "none" or hdata.linewidth == 0.0
                            else mcolors.to_rgba(hdata.edgecolor, alpha=hdata.edgealpha)
                        ),
                        zorder=hdata.zorder,
                        label=hdata.name,
                    )
                    continue

                hist_edges = bin_edges[:-1].copy()
                if n_bins_per_bar > 1:
                    hist_edges += (bin_widths / n_bins_per_bar) * i

                self._ax.bar(
                    hist_edges,
                    hist_heights,
                    width=bin_widths / n_bins_per_bar,
                    bottom=hist_bottoms,
                    align=self._bin_alignment,
                    facecolor=mcolors.to_rgba(hdata.facecolor, alpha=hdata.facealpha),
                    edgecolor=(
                        "none"
                        if hdata.edgecolor == "none" or hdata.linewidth == 0.0
                        else mcolors.to_rgba(hdata.edgecolor, alpha=hdata.edgealpha)
                    ),
                    linewidth=hdata.linewidth,
                    zorder=hdata.zorder,
                )

                if histtype == "stack":
                    hist_bottoms += hist_heights

    def _draw_points(self) -> None:
        """Draw the points on the histogram plot.

        Computes the maximum heights of the histogram bars to position the points
        just above the bars, taking into account marker size and edge width.
        """
        bin_edges = self._compute_bins()
        max_heights = np.zeros(len(bin_edges) - 1)

        def marker_clearance(y_top, markersize_pt, markeredgewidth_pt, marker, pad_pt=0.0):
            ms = MarkerStyle(marker)
            path = ms.get_path().transformed(ms.get_transform())

            verts = np.asarray(path.vertices, dtype=float)
            ys = verts[:, 1]
            y0 = float(ys.min())
            y1 = float(ys.max())

            height_u = y1 - y0
            if height_u == 0.0:
                clearance_pt = 0.5 * markersize_pt + 0.5 * markeredgewidth_pt + pad_pt
            else:
                pt_per_u = markersize_pt / height_u
                bottom_pt = y0 * pt_per_u
                clearance_pt = (-bottom_pt) + 0.5 * markeredgewidth_pt + pad_pt

            # points -> pixels
            px = clearance_pt * self._ax.figure.dpi / 72.0

            # pixels -> data-units at y_top
            p = self._ax.transData.transform((0.0, y_top))
            y2 = self._ax.transData.inverted().transform(p + np.array([0.0, px]))[1]
            return y2 - y_top

        for histtype, histlist in self._hist_data_dict.items():
            hist_bottoms = np.zeros(len(bin_edges) - 1)

            for i, hdata in enumerate(histlist):
                hist_heights = np.histogram(
                    hdata.values,
                    bins=bin_edges,
                    weights=hdata.weights,
                    density=self.as_denisty_plot,
                )[0]

                if histtype == "stack":
                    hist_bottoms += hist_heights
                    hist_top = hist_bottoms
                else:
                    hist_top = hist_heights

                max_heights = np.maximum(max_heights, hist_top)

        for pointlist in self._histpointlist_list:
            x_positions = np.array(pointlist.values)
            y_positions = []
            visited_indexes = set()
            for i, val in enumerate(pointlist.values):
                bin_idx = np.searchsorted(bin_edges, val, side="right") - 1
                in_range = 0 <= bin_idx < len(max_heights)
                if not in_range:
                    dy = marker_clearance(
                        0.0,
                        pointlist.point_data.markersize,
                        pointlist.point_data.markeredgewidth,
                        pointlist.point_data.marker,
                    )
                    y_positions.append(pointlist.y_offset + dy)
                else:
                    dy = marker_clearance(
                        max_heights[bin_idx],
                        pointlist.point_data.markersize,
                        pointlist.point_data.markeredgewidth,
                        pointlist.point_data.marker,
                    )
                    offset = dy
                    if bin_idx not in visited_indexes:
                        visited_indexes.add(bin_idx)
                        offset += pointlist.y_offset

                    y_positions.append(max_heights[bin_idx] + offset)

                    max_heights[bin_idx] += offset + dy
                    if pointlist.centered:
                        centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
                        x_positions[i] = centers[bin_idx]

            self._ax.plot(
                x_positions,
                y_positions,
                linestyle="",
                marker=pointlist.point_data.marker,
                markersize=pointlist.point_data.markersize,
                markerfacecolor=mcolors.to_rgba(
                    pointlist.point_data.markerfacecolor,
                    alpha=pointlist.point_data.markerfacealpha,
                ),
                markeredgecolor=mcolors.to_rgba(
                    pointlist.point_data.markeredgecolor,
                    alpha=pointlist.point_data.markeredgealpha,
                ),
                markeredgewidth=pointlist.point_data.markeredgewidth,
                zorder=pointlist.point_data.zorder,
                label=pointlist.name,
            )

    def _build_plot(self) -> None:
        """Build the histogram plot."""
        if sum(len(lst) for lst in self._hist_data_dict.values()) == 0:
            raise ValueError("No histogram sets added yet.")

        self._ax.clear()

        self._draw_verticals()
        self._draw_horizontals()

        if self.xlabel is not None:
            self._ax.set_xlabel(self.xlabel)
        if self.ylabel is not None:
            self._ax.set_ylabel(self.ylabel)
        if self.title is not None:
            self._ax.set_title(self.title)

        self._ax.grid(self.grid)

        self._draw_histograms()
        self._set_x_axis()
        self._set_y_axis()

        self._draw_points()

        if self.include_legend:
            self._ax.legend(
                handles=self._legend_handles,
                **self._legend_options.to_dict(),
            )

    def _get_histogram_legend_handles(self) -> list[Any]:
        """Get legend handles for the histogram sets."""
        handles: list[Any] = []
        for hdata in chain(*self._hist_data_dict.values()):
            handles.append(
                Patch(
                    facecolor=mcolors.to_rgba(hdata.facecolor, alpha=hdata.facealpha),
                    edgecolor=(
                        "none"
                        if hdata.edgecolor == "none" or hdata.linewidth == 0.0
                        else mcolors.to_rgba(hdata.edgecolor, alpha=hdata.edgealpha)
                    ),
                    linewidth=hdata.linewidth,
                    label=hdata.name,
                )
            )
        return handles

    def _get_pointset_legend_handles(self) -> list[Any]:
        """Generate legend handles for point sets.

        Returns:
            list[Any]: A list of legend handles for the point sets.
        """
        handles: list[Any] = []

        for histpoint in self._histpointlist_list:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="none",
                    label=histpoint.name,
                    **histpoint.point_data.to_mpl_settings_dict(),
                )
            )

        return handles

    @property
    def _legend_handles(self) -> list[Any]:
        """Get all legend handles for the plot."""
        handles: list[Any] = []
        handles.extend(self._get_histogram_legend_handles())
        handles.extend(self._get_named_line_legend_handles())
        handles.extend(self._get_named_band_legend_handles())
        handles.extend(self._get_pointset_legend_handles())
        return handles
