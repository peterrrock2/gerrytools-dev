from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from numbers import Real
from typing import Any

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from gerrytools.colors import resolve_color_and_alpha
from gerrytools.logging import get_logger
from gerrytools.plotting._gerryplot_dataclasses import LineData, PointSetData
from gerrytools.plotting._gerryplot_to_mpl_option_dataclasses import PointMarkerOptions
from gerrytools.plotting.gerryplot import GerryPlotBase
from gerrytools.typing import Color

logger = get_logger(__name__)


@dataclass(frozen=True)
class ViolinPlotSetData:
    """A dataclass representing a set of violinplots to be plotted on a violinplot figure.

    Attributes:
        name (str): The name of the violinplot set.
        scores_dict (dict[str, list[float]]): A dictionary mapping labels to lists of scores.
        facecolor (Color): The face color of the violinplots.
        facealpha (float | None): The alpha transparency of the violinplots.
        edgecolor (Color): The edge color of the violinplots.
        edgealpha (float | None): The alpha transparency of the violinplot edges.
        edgewidth (float): The edgewidth of the violinplot edges.
        zorder (int): The z-order of the violinplots.
    """

    name: str
    scores_dict: dict[str, list[float]]
    facecolor: Color
    facealpha: float | None = None
    edgecolor: Color = "black"
    edgealpha: float | None = None
    edgewidth: float = 0.8
    zorder: int = 1

    def __post_init__(self) -> None:
        lw = float(self.edgewidth)
        if not math.isfinite(lw):
            raise ValueError("edgewidth must be a finite number")
        if lw < 0:
            raise ValueError("edgewidth must be nonnegative")
        object.__setattr__(self, "edgewidth", lw)

        resolved_facecolor, resolved_alpha = resolve_color_and_alpha(
            self.facecolor,
            alpha=self.facealpha,
            allow_none=True,
            field="facecolor",
            owner=f"ViolinPlotSetData {self.name}",
            logger=logger,
        )
        object.__setattr__(self, "facecolor", resolved_facecolor)
        object.__setattr__(self, "facealpha", resolved_alpha)

        resolved_edgecolor, resolved_edgealpha = resolve_color_and_alpha(
            self.edgecolor,
            alpha=self.edgealpha,
            allow_none=True,
            field="edgecolor",
            owner=f"ViolinPlotSetData {self.name}",
            logger=logger,
        )
        object.__setattr__(self, "edgecolor", resolved_edgecolor)
        object.__setattr__(self, "edgealpha", resolved_edgealpha)

        if resolved_edgecolor.lower() == "none" and lw > 0:
            logger.log(
                level=logging.DEBUG,
                msg=(
                    f"For ViolinPlotSetData {self.name}: edgecolor is 'none' but "
                    f"edgewidth is {lw}>0; setting edgewidth to 0."
                ),
            )
            lw = 0.0

        object.__setattr__(self, "edgewidth", lw)
        object.__setattr__(self, "zorder", int(self.zorder))


class ViolinPlot(GerryPlotBase):
    """A class for creating violinplot comparison figures with multiple violinplot sets and point sets
    representing a distribution of scores across multiple categories.

    Typical usage:
        violinplot = ViolinPlot(figure_size=(12, 8), dpi=300)
        violinplot.add_violinplot_datasets(
            scores={'Category A': [1, 2, 3], 'Category B': [2, 3, 4]},
            name='Model 1',
            facecolor='blue',
            edgecolor='black'
        )
        violinplot.add_violinplot_datasets(
            scores={'Category A': [2, 3, 4], 'Category B': [3, 4, 5]},
            name='Model 2',
            facecolor='green',
            edgecolor='black'
        )
        violinplot.add_pointset(
            values={'Category A': 2.5, 'Category B': 3.5},
            name='Mean Scores',
            facecolor='red',
            marker='o'
        )
        violinplot.show()
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
        violinplot_width_scale: float = 0.8,
        violinplot_group_width: float = 0.7,
        include_violinplot_group_vlines: bool = False,
    ) -> None:
        """Initialize a ViolinPlot instance.

        Args:
            figure_size (tuple[float, float], optional): The size of the figure in inches.
                Defaults to (10, 6).
            dpi (int, optional): The dots per inch (DPI) of the figure. Defaults to 300.
            include_legend (bool, optional): Whether to include a legend in the plot.
                Defaults to True.
            xlabel (str | None, optional): The label for the x-axis. Defaults to None.
            ylabel (str | None, optional): The label for the y-axis. Defaults to None.
            title (str | None, optional): The title of the plot. Defaults to None.
            violinplot_group_width (float, optional): The width of each violinplot group.
                Defaults to 0.7
            violinplot_width_scale (float, optional): The scaling factor for violinplot widths
                within each group. Defaults to 0.8.
            include_violinplot_group_vlines (bool, optional): Whether to include vertical lines
                at the center of the violinplot groups. Defaults to False.
        """
        super().__init__(
            figure_size=figure_size,
            dpi=dpi,
            include_legend=include_legend,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
        )

        self._violinplot_data_list: list[ViolinPlotSetData] = []
        self._pointset_data_list: list[PointSetData] = []
        self._labels: list[str] | None = None

        if violinplot_group_width <= 0:
            raise ValueError("violinplot_group_width must be positive")
        if violinplot_group_width > 1.0:
            raise ValueError("violinplot_group_width must be <= 1.0 when centers are integers.")
        if not 0.0 < violinplot_width_scale <= 1.0:
            raise ValueError("violinplot_width_scale must be in (0.0, 1.0].")

        self.violinplot_group_width = float(violinplot_group_width)
        self.violinplot_width_scale = float(violinplot_width_scale)

        self._include_violinplot_group_vlines = include_violinplot_group_vlines
        self._violinplot_group_vline_settings = LineData(
            values=float("inf"),  # placeholder
            linecolor="#cccccc",
            linealpha=1.0,
            linestyle="-",
            linewidth=0.8,
            zorder=-3,
        )

    @staticmethod
    def _convert_violinplot_data_to_dictionary(
        scores: dict[str, list[float]] | list[float] | list[list[float]] | pd.DataFrame,
        scores_labels: list[str] | None = None,
    ) -> dict[str, list[float]]:
        """Convert violinplot input to a dictionary mapping labels to score lists.

        Args:
            scores (dict[str, list[float]] | list[float] | list[list[float]] | pd.DataFrame):
                The violinplot scores. Can be a dictionary mapping labels to score lists,
                a list of score lists, or a DataFrame where each column represents a label.
            scores_labels (list[str] | None, optional): The labels for the scores if
                scores is a list or list of lists. Defaults to None.

        Returns:
            dict[str, list[float]]: A dictionary mapping labels to score lists.
        """
        if isinstance(scores, dict):
            return {str(k): list(v) for k, v in scores.items()}

        if isinstance(scores, pd.DataFrame):
            return {str(col): scores[col].dropna().tolist() for col in scores.columns}

        if isinstance(scores, list):
            if scores_labels is None:
                raise ValueError(
                    "When providing lists of scores, please also provide labels for each list."
                )

            if len(scores) == 0:
                raise ValueError("scores is empty; provide at least one score list.")

            first = scores[0]

            def _is_scalar(x: Any) -> bool:
                return isinstance(x, (str, bytes, Real, np.generic))

            def _is_score_series(x: Any) -> bool:
                # Accept common “list of numbers” containers.
                if _is_scalar(x):
                    return False
                return isinstance(x, (list, tuple, np.ndarray, pd.Series))

            scores_list_of_lists = scores if _is_score_series(first) else [scores]

            if len(scores_labels) != len(scores_list_of_lists):
                raise ValueError(
                    f"scores_labels has length {len(scores_labels)} but you provided "
                    f"{len(scores_list_of_lists)} score lists."
                )

            return {
                label: list(score_list)
                for label, score_list in zip(scores_labels, scores_list_of_lists, strict=True)
            }

        raise TypeError(
            "Scores must be a dict[str, list[float]], list[float], list[list[float]], "
            "or pd.DataFrame."
        )

    def add_violinplot_datasets(
        self,
        scores: dict[str, list[float]] | list[float] | list[list[float]] | pd.DataFrame,
        *,
        scores_labels: list[str] | None = None,
        name: str | None = None,
        facecolor: Color = "denim",
        facealpha: float | None = None,
        edgecolor: Color = "black",
        edgealpha: float | None = None,
        edgewidth: float = 0.8,
        add_extra_labels: bool = False,
        zorder: int = 1,
    ) -> None:
        """Add a set of violinplots to the figure.

        Args:
            scores (dict[str, list[float]] | list[float] | list[list[float]] | pd.DataFrame):
                The scores for the violinplots. Can be a dictionary mapping labels to score lists,
                a list of score lists, or a DataFrame where each column represents a label.
            scores_labels (list[str] | None, optional): The labels for the scores if
                scores is a list or list of lists. Defaults to None.
            name (str | None, optional): The name of the violinplot set. Defaults to None.
            facecolor (Color, optional): The face color of the violinplots. Defaults to "denim".
            facealpha (float | None, optional): The alpha transparency of the violinplots.
                Defaults to None.
            edgecolor (Color, optional): The edge color of the violinplots. Defaults to "black".
            edgealpha (float | None, optional): The alpha transparency of the violinplot edges.
                Defaults to None.
            edgewidth (float, optional): The edgewidth of the violinplot edges. Defaults to 0.8.
            add_extra_labels (bool, optional): Whether to allow adding new labels.
                Defaults to False.
            zorder (int, optional): The z-order of the violinplots. Defaults to 1.

        Raises:
            ValueError: If the labels of the incoming violinplot set do not match the existing labels
                and add_extra_labels is False.

        Returns:
            None
        """

        scores_dict = self._convert_violinplot_data_to_dictionary(scores, scores_labels)

        if self._labels is None:
            self._labels = list(scores_dict.keys())
        else:
            incoming = list(scores_dict.keys())
            if incoming != self._labels:
                if not add_extra_labels:
                    raise ValueError(
                        "All sets must use the same labels in the same order.\n"
                        f"Expected: {self._labels}\nGot:      {incoming}\n"
                        "If you want to allow for additional labels, set add_extra_labels=True."
                    )
                label_set = list(dict.fromkeys(self._labels + incoming))
                self._labels = label_set

        set_name = name or f"Set {len(self._violinplot_data_list) + 1}"
        self._violinplot_data_list.append(
            ViolinPlotSetData(
                scores_dict=scores_dict,
                name=set_name,
                facecolor=facecolor,
                facealpha=facealpha,
                edgecolor=edgecolor,
                edgealpha=edgealpha,
                edgewidth=edgewidth,
                zorder=zorder,
            )
        )

    def _convert_pointset_to_dict(
        self,
        values: dict[str, float] | list[float] | pd.Series | pd.DataFrame,
        labels: list[str] | None = None,
        *,
        column: str | None = None,
    ) -> dict[str, float]:
        """Convert pointset input to a dictionary mapping labels to float values.

        Args:
            values (dict[str, float] | list[float] | pd.Series | pd.DataFrame):
                The pointset values. Can be a dictionary mapping labels to values,
                a list of values, a Series, or a DataFrame.

            labels (list[str] | None, optional): The labels for the pointset values
                if values is a list. Defaults to None.
            column (str | None, optional): The column name to use if values is a DataFrame.

        Returns:
            dict[str, float]: A dictionary mapping labels to float values.


        Raises:
            ValueError: If values is a DataFrame and column is None and the DataFrame has
                more than one column.
            ValueError: If the labels of the incoming point set do not match the existing labels
                when values is a list and labels is None.
            ValueError: If the length of values does not match the length of labels
        """
        if isinstance(values, dict):
            return {str(k): float(v) for k, v in values.items()}

        if isinstance(values, pd.Series):
            return {str(k): float(v) for k, v in values.items()}

        if isinstance(values, pd.DataFrame):
            if column is None:
                if values.shape[1] != 1:
                    raise ValueError(
                        "DataFrame pointset input must have exactly one column, or pass column=..."
                    )
                ser = values.iloc[:, 0]
            else:
                ser = values[column]
            return {str(k): float(v) for k, v in ser.items()}

        vals = list(values)
        if labels is None:
            if self._labels is None:
                raise ValueError(
                    "For list pointset input, provide labels=... (or add violinplots first to "
                    "define labels)."
                )
            labels = self._labels

        if len(vals) != len(labels):
            raise ValueError(
                f"Point set values length {len(vals)} does not match labels length {len(labels)}."
            )
        return dict(zip(labels, map(float, vals), strict=True))

    def add_pointset(
        self,
        values: dict[str, float] | list[float] | pd.Series | pd.DataFrame,
        *,
        labels: list[str] | None = None,
        column: str | None = None,
        name: str | None = None,
        facecolor: Color = "black",
        facealpha: float | None = None,
        marker: str = "o",
        markersize: float = 7.0,
        markeredgecolor: Color = "black",
        markeredgealpha: float | None = None,
        markeredgewidth: float = 0.8,
        x_offset: float | None = None,
        zorder: int = 2,
        add_extra_labels: bool = False,
    ) -> None:
        """Add a set of points to the figure.

        Args:
            values (dict[str, float] | list[float] | pd.Series | pd.DataFrame):
                The pointset values. Can be a dictionary mapping labels to values,
                a list of values, a Series, or a DataFrame.
            labels (list[str] | None, optional): The labels for the point values
                if values is a list. Defaults to None.
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
            x_offset (float | None, optional): An absolute x-offset from category center.
                Defaults to None.
            zorder (int, optional): The z-order of the points. Defaults to 2.
            add_extra_labels (bool, optional): Whether to allow adding new labels.
                Defaults to False.

        Raises:
            ValueError: If the labels of the incoming point set do not match the existing labels
                and add_extra_labels is False.

        Returns:
            None
        """
        values_dict = self._convert_pointset_to_dict(values, labels, column=column)

        incoming = list(values_dict.keys())
        if self._labels is None:
            self._labels = incoming
        else:
            if incoming != self._labels:
                if not add_extra_labels:
                    raise ValueError(
                        "point set labels must match existing labels in the same order.\n"
                        f"Expected: {self._labels}\nGot:      {incoming}\n"
                        "If you want to allow for additional labels, set add_extra_labels=True."
                    )
                self._labels = list(dict.fromkeys(self._labels + incoming))

        set_name = name or f"Point Set {len(self._pointset_data_list) + 1}"
        self._pointset_data_list.append(
            PointSetData(
                name=set_name,
                values_dict=values_dict,
                point_data=PointMarkerOptions(
                    markerfacecolor=facecolor,
                    markerfacealpha=facealpha,
                    marker=marker,
                    markersize=markersize,
                    markeredgecolor=markeredgecolor,
                    markeredgealpha=markeredgealpha,
                    markeredgewidth=markeredgewidth,
                    zorder=zorder,
                ),
                x_offset=x_offset,
            )
        )

    def remove_violinplot_group_vlines(self) -> None:
        """Remove vertical lines at the center of violinplot groups."""
        self._include_violinplot_group_vlines = False

    def update_violinplot_group_vline_settings(
        self,
        *,
        linecolor: Color = "#cccccc",
        linealpha: float = 1.0,
        linestyle: str = "-",
        linewidth: float = 0.8,
        zorder: int = -3,
    ) -> None:
        """Update the settings for vertical lines at the center of violinplot groups.

        Args:
            linecolor (Color, optional): The color of the vertical lines. Defaults to "#cccccc".
            linealpha (float, optional): The alpha transparency of the vertical lines.
                Defaults to 1.0.
            linestyle (str, optional): The linestyle of the vertical lines. Defaults to "-".
            linewidth (float, optional): The width of the vertical lines. Defaults to 0.8.
            zorder (int, optional): The z-order of the vertical lines. Defaults to -3.

        Returns:
            None
        """
        self._include_violinplot_group_vlines = True
        self._violinplot_group_vline_settings = LineData(
            values=float("inf"),  # placeholder
            linecolor=linecolor,
            linealpha=linealpha,
            linestyle=linestyle,
            linewidth=linewidth,
            zorder=zorder,
        )

    def clear_vertical_lines_and_bands(self) -> None:
        """Clear all vertical lines and bands from the figure."""
        self._include_violinplot_group_vlines = False
        self._vertical_lines.clear()
        self._vertical_bands.clear()

    def _default_x_tick_locations(self) -> list[float] | None:
        """Get default x-tick locations at the center of each violinplot group."""
        return list(self._violinplot_centers)

    def _default_x_tick_labels(self, tick_locations: list[float]) -> list[str] | None:
        """Get default x-tick labels for the violinplot categories."""
        if self._labels is None:
            return None
        # Only apply category labels when lengths match; if the user overrides locations to
        # something else, leave labels alone unless they explicitly set them.
        if len(tick_locations) == len(self._labels):
            return list(self._labels)
        return None

    def _draw_violinplot_group_vlines(self) -> None:
        """Draw vertical lines at the center of violinplot groups."""
        for x in self._violinplot_centers:
            self._ax.axvline(
                x,
                color=mcolors.to_rgba(
                    self._violinplot_group_vline_settings.linecolor,
                    alpha=self._violinplot_group_vline_settings.linealpha,
                ),
                linestyle=self._violinplot_group_vline_settings.linestyle,
                linewidth=self._violinplot_group_vline_settings.linewidth,
                zorder=self._violinplot_group_vline_settings.zorder,
            )

    @property
    def _violinplot_centers(self) -> np.ndarray:
        """Calculate the x-axis centers for each violinplot category."""
        if self._labels is None:
            return np.array([])

        n_categories = len(self._labels)
        centers = 1.0 + np.arange(n_categories, dtype=float)
        return centers

    def _draw_pointset(self, centers: np.ndarray, *, span: float | None = None) -> None:
        """Draw pointset on the plot.

        Args:
            centers (np.ndarray): The x-axis centers for each violinplot category.
            span (float | None, optional): The total span for auto-offsetting pointsets.
                Defaults to None, which uses 80% of violinplot_group_width or 0.8, whichever is
                smaller.

        Returns:
            None
        """
        if len(self._pointset_data_list) == 0 or self._labels is None:
            return

        n = len(self._pointset_data_list)

        # auto-offsets to reduce overlap between multiple point sets (still “lined up” per label)
        if span is None:
            span = min(self.violinplot_group_width * 0.8, 0.8)

        auto_offsets = (np.arange(n) - (n - 1) / 2.0) * (span / max(n, 1))

        for i, sdata in enumerate(self._pointset_data_list):
            offset = float(sdata.x_offset) if sdata.x_offset is not None else float(auto_offsets[i])
            xs = centers + offset

            ys = np.array([sdata.values_dict.get(lab, np.nan) for lab in self._labels], dtype=float)
            mask = np.isfinite(ys)
            if not np.any(mask):
                continue
            self._ax.plot(
                xs[mask],
                ys[mask],
                linestyle="none",
                clip_on=True,
                rasterized=True,
                **sdata.point_data.to_mpl_settings_dict(),
            )

    def _draw_violinplot(self) -> None:
        """Draw the violinplots on the plot."""
        n_violinplot_sets = len(self._violinplot_data_list)

        # Category centers on x axis
        centers = self._violinplot_centers

        # Width and offsets for each set within a category
        violin_width = self.violinplot_group_width / n_violinplot_sets
        # center the offsets around the category center
        offsets = (np.arange(n_violinplot_sets) - (n_violinplot_sets - 1) / 2.0) * violin_width
        widths = violin_width * self.violinplot_width_scale

        for k, violinplot_data in enumerate(self._violinplot_data_list):
            pos_k_all = centers + offsets[k]
            data_k: list[list[float]] = []
            pos_k: list[float] = []

            if self._labels is None:
                continue

            for lab, x in zip(self._labels, pos_k_all, strict=True):
                vals = violinplot_data.scores_dict.get(lab, [])
                if vals is None or len(vals) == 0:
                    continue
                data_k.append(list(vals))
                pos_k.append(float(x))

            if len(data_k) == 0:
                continue

            vp = self._ax.violinplot(
                data_k,
                positions=pos_k,
                widths=widths,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )

            for patch in vp["bodies"]:
                patch.set_alpha(None)
                patch.set_facecolor(
                    mcolors.to_rgba(violinplot_data.facecolor, alpha=violinplot_data.facealpha)
                )
                patch.set_linewidth(violinplot_data.edgewidth)
                patch.set_edgecolor(
                    mcolors.to_rgba(violinplot_data.edgecolor, alpha=violinplot_data.edgealpha)
                )
                patch.set_zorder(violinplot_data.zorder)

    def _build_plot(self) -> None:
        """Build the violinplot figure."""
        if self._labels is None or len(self._labels) == 0:
            raise ValueError("No labels defined yet.")

        if len(self._violinplot_data_list) == 0:
            raise ValueError("No violinplot sets added yet.")

        self._draw_violinplot()
        self._draw_pointset(self._violinplot_centers)
        if self._include_violinplot_group_vlines:
            self._draw_violinplot_group_vlines()

    def _get_violinplot_legend_handles(self) -> list[Any]:
        """Generate legend handles for violinplot sets.

        Returns:
            list[Any]: A list of legend handles for the violinplot sets.
        """
        handles: list[Any] = []

        for violinplot_data in self._violinplot_data_list:
            handles.append(
                Patch(
                    facecolor=mcolors.to_rgba(
                        violinplot_data.facecolor, alpha=violinplot_data.facealpha
                    ),
                    edgecolor=mcolors.to_rgba(
                        violinplot_data.edgecolor, alpha=violinplot_data.edgealpha
                    ),
                    label=violinplot_data.name,
                )
            )

        return handles

    def _get_pointset_legend_handles(self) -> list[Any]:
        """Generate legend handles for point sets.

        Returns:
            list[Any]: A list of legend handles for the point sets.
        """
        handles: list[Any] = []

        for sdata in self._pointset_data_list:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="none",
                    label=sdata.name,
                    **sdata.point_data.to_mpl_settings_dict(),
                )
            )

        return handles

    @property
    def _legend_handles(self) -> list[Any]:
        """Generated legend handles for violinplot and point sets."""
        handles: list[Any] = []

        handles.extend(self._get_violinplot_legend_handles())
        handles.extend(self._get_pointset_legend_handles())
        handles.extend(self._get_named_line_legend_handles())
        handles.extend(self._get_named_band_legend_handles())

        return handles
