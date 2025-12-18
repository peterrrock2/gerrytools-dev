from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from gerrytools.logging import get_logger
from gerrytools.plotting.colors import resolve_color_and_alpha
from gerrytools.plotting.gerryplot import (
    GerryPlotBase,
    LineData,
    ScatterPointSettings,
)
from gerrytools.typing import Color

logger = get_logger(__name__)


@dataclass(frozen=True)
class ScatterSetData:
    """A dataclass representing a set of scatter points to be plotted on a boxplot figure.

    Attributes:
        name (str): The name of the scatter set.
        values_dict (dict[str, float]): A dictionary mapping labels to scatter point values.
        point_data (ScatterPointSettings): The settings for the scatter points.
        x_offset (float | None): An optional absolute x-offset from category center.
    """

    name: str
    values_dict: dict[str, float]  # one value per label
    point_data: ScatterPointSettings
    x_offset: float | None = None  # optional absolute x-offset from category center


@dataclass(frozen=True)
class BoxPlotSetData:
    """A dataclass representing a set of boxplots to be plotted on a boxplot figure.

    Attributes:
        name (str): The name of the boxplot set.
        scores_dict (dict[str, list[float]]): A dictionary mapping labels to lists of scores.
        face_color (Color): The face color of the boxplots.
        alpha (float | None): The alpha transparency of the boxplots.
        percentiles (tuple[float, float]): The percentiles for the whiskers.
        showfliers (bool): Whether to show outliers.
        fliersettings (ScatterPointSettings): The settings for outlier points.
        edge_color (Color): The edge color of the boxplots.
        edge_alpha (float | None): The alpha transparency of the boxplot edges.
        linewidth (float): The linewidth of the boxplot edges.
        zorder (int): The z-order of the boxplots.
    """

    name: str
    scores_dict: dict[str, list[float]]
    face_color: Color
    alpha: float | None = None
    percentiles: tuple[float, float] = (1, 99)
    showfliers: bool = False
    fliersettings: ScatterPointSettings = field(default_factory=ScatterPointSettings)
    edge_color: Color = "black"
    edge_alpha: float | None = None
    linewidth: float = 0.8
    zorder: int = 2

    def __post_init__(self) -> None:
        lo, hi = self.percentiles
        lo = float(lo)
        hi = float(hi)
        if not (0.0 <= lo <= 100.0 and 0.0 <= hi <= 100.0):
            raise ValueError("percentiles must be within [0, 100].")
        if not (lo < hi):
            raise ValueError("percentiles must satisfy low < high.")

        lw = float(self.linewidth)
        if not math.isfinite(lw):
            raise ValueError("linewidth must be a finite number")
        if lw < 0:
            raise ValueError("linewidth must be nonnegative")
        object.__setattr__(self, "linewidth", lw)

        resolved_face_color, resolved_alpha = resolve_color_and_alpha(
            self.face_color,
            alpha=self.alpha,
            allow_none=True,
            field="face_color",
            owner=f"BoxPlotSetData {self.name}",
            logger=logger,
        )
        object.__setattr__(self, "face_color", resolved_face_color)
        object.__setattr__(self, "alpha", resolved_alpha)

        resolved_edge_color, resolved_edge_alpha = resolve_color_and_alpha(
            self.edge_color,
            alpha=self.edge_alpha,
            allow_none=True,
            field="edge_color",
            owner=f"BoxPlotSetData {self.name}",
            logger=logger,
        )
        object.__setattr__(self, "edge_color", resolved_edge_color)
        object.__setattr__(self, "edge_alpha", resolved_edge_alpha)

        if resolved_edge_color.lower() == "none" and lw > 0:
            logger.log(
                level=logging.DEBUG,
                msg=(
                    f"For BoxPlotSetData {self.name}: edge_color is 'none' but "
                    f"linewidth is {lw}>0; setting linewidth to 0."
                ),
            )
            lw = 0.0

        object.__setattr__(self, "linewidth", lw)
        object.__setattr__(self, "zorder", int(self.zorder))


class BoxPlot(GerryPlotBase):
    """A class for creating boxplot comparison figures with multiple boxplot sets and scatter sets
    representing a distribution of scores across multiple categories.
    """

    def __init__(
        self,
        figure_size: tuple[float, float] = (10, 6),
        dpi: int = 300,
        *,
        boxplot_width_scale: float = 0.8,
        boxplot_group_width: float = 0.7,
        include_legend: bool = False,
        include_boxplot_group_vlines: bool = True,
    ) -> None:
        """Initialize a BoxPlotComparison instance.

        Args:
            figure_size (tuple[float, float], optional): The size of the figure in inches.
                Defaults to (10, 6).
            dpi (int, optional): The dots per inch (DPI) of the figure. Defaults to 300.

        Kwargs:
            boxplot_group_width (float, optional): The width of each boxplot group.
                Defaults to 0.7
            boxplot_width_scale (float, optional): The scaling factor for boxplot widths
                within each group. Defaults to 0.8.
            include_legend (bool, optional): Whether to include a legend in the plot.
                Defaults to False.
            include_boxplot_group_vlines (bool, optional): Whether to include vertical lines
                at the center of the boxplot groups. Defaults to True.
        """
        super().__init__(figure_size=figure_size, dpi=dpi, include_legend=include_legend)

        self._boxplot_data_list: list[BoxPlotSetData] = []
        self._scatter_data_list: list[ScatterSetData] = []
        self._labels: list[str] | None = None

        if boxplot_group_width <= 0:
            raise ValueError("boxplot_group_width must be positive")
        if boxplot_group_width > 1.0:
            raise ValueError("boxplot_group_width must be <= 1.0 when centers are integers.")
        if not 0.0 < boxplot_width_scale <= 1.0:
            raise ValueError("boxplot_width_scale must be in (0.0, 1.0].")

        self.boxplot_group_width = float(boxplot_group_width)
        self.boxplot_width_scale = float(boxplot_width_scale)

        self._include_boxplot_group_vlines = include_boxplot_group_vlines
        self._boxplot_group_vline_settings = LineData(
            value=float("inf"),  # placeholder
            linecolor="#cccccc",
            linealpha=1.0,
            linestyle="-",
            linewidth=0.8,
            zorder=-3,
        )

    @staticmethod
    def _convert_boxplot_data_to_dictionary(
        scores: dict[str, list[float]] | list[float] | list[list[float]] | pd.DataFrame,
        scores_labels: list[str] | None = None,
    ) -> dict[str, list[float]]:
        """Convert boxplot input to a dictionary mapping labels to score lists.

        Args:
            scores (dict[str, list[float]] | list[float] | list[list[float]] | pd.DataFrame):
                The boxplot scores. Can be a dictionary mapping labels to score lists,
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
                scores_list_of_lists = [scores]
            else:
                first = scores[0]
                is_series = isinstance(first, Sequence) and not isinstance(first, (str, bytes))
                scores_list_of_lists = scores if is_series else [scores]

            if len(scores_labels) != len(scores_list_of_lists):
                raise ValueError(
                    f"scores_labels has length {len(scores_labels)} but you provided "
                    f"{len(scores_list_of_lists)} score lists."
                )

            return {
                label: score_list
                for label, score_list in zip(scores_labels, scores_list_of_lists, strict=True)
            }

        raise TypeError(
            "Scores must be a dict[str, list[float]], list[float], list[list[float]], "
            "or pd.DataFrame."
        )

    def add_boxplot_set(
        self,
        scores: dict[str, list[float]] | list[float] | list[list[float]] | pd.DataFrame,
        *,
        name: str | None = None,
        face_color: Color = "denim",
        scores_labels: list[str] | None = None,
        percentiles: tuple[float, float] = (1, 99),
        showfliers: bool = False,
        fliersettings: ScatterPointSettings | None = None,
        alpha: float | None = None,
        edge_color: Color = "black",
        edge_alpha: float | None = None,
        add_extra_labels: bool = False,
        zorder: int = 2,
    ) -> None:
        """Add a set of boxplots to the figure.

        Args:
            scores (dict[str, list[float]] | list[float] | list[list[float]] | pd.DataFrame):
                The scores for the boxplots. Can be a dictionary mapping labels to score lists,
                a list of score lists, or a DataFrame where each column represents a label.

        Kwargs:
            name (str | None, optional): The name of the boxplot set. Defaults to None.
            face_color (Color, optional): The face color of the boxplots. Defaults to "denim".
            scores_labels (list[str] | None, optional): The labels for the scores if
                scores is a list or list of lists. Defaults to None.
            percentiles (tuple[float, float], optional): The percentiles for the whiskers.
                Defaults to (1, 99).
            showfliers (bool, optional): Whether to show outliers. Defaults to False.
            fliersettings (ScatterPointSettings | None, optional): The settings for outlier points.
                Defaults to None.
            alpha (float | None, optional): The alpha transparency of the boxplots.
                Defaults to None.
            edge_color (Color, optional): The edge color of the boxplots. Defaults to "black".
            edge_alpha (float | None, optional): The alpha transparency of the boxplot edges.
                Defaults to None.
            add_extra_labels (bool, optional): Whether to allow adding new labels.
                Defaults to False.
            zorder (int, optional): The z-order of the boxplots. Defaults to 2.

        Raises:
            ValueError: If the labels of the incoming boxplot set do not match the existing labels
                and add_extra_labels is False.

        Returns:
            None
        """
        if fliersettings is None:
            fliersettings = ScatterPointSettings()

        scores_dict = self._convert_boxplot_data_to_dictionary(scores, scores_labels)

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

        set_name = name or f"Set {len(self._boxplot_data_list) + 1}"
        self._boxplot_data_list.append(
            BoxPlotSetData(
                scores_dict=scores_dict,
                name=set_name,
                percentiles=percentiles,
                face_color=face_color,
                showfliers=showfliers,
                fliersettings=fliersettings,
                alpha=alpha,
                edge_color=edge_color,
                edge_alpha=edge_alpha,
                zorder=zorder,
            )
        )

    def _convert_scatter_to_dict(
        self,
        values: dict[str, float] | list[float] | pd.Series | pd.DataFrame,
        labels: list[str] | None = None,
        *,
        column: str | None = None,
    ) -> dict[str, float]:
        """Convert scatter input to a dictionary mapping labels to float values.

        Args:
            values (dict[str, float] | list[float] | pd.Series | pd.DataFrame):
                The scatter values. Can be a dictionary mapping labels to values,
                a list of values, a Series, or a DataFrame.

            labels (list[str] | None, optional): The labels for the scatter values
                if values is a list. Defaults to None.

        Kwargs:
            column (str | None, optional): The column name to use if values is a DataFrame.

        Returns:
            dict[str, float]: A dictionary mapping labels to float values.


        Raises:
            ValueError: If values is a DataFrame and column is None and the DataFrame has
                more than one column.
            ValueError: If the labels of the incoming scatter set do not match the existing labels
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
                        "DataFrame scatter input must have exactly one column, or pass column=..."
                    )
                ser = values.iloc[:, 0]
            else:
                ser = values[column]
            return {str(k): float(v) for k, v in ser.items()}

        vals = list(values)
        if labels is None:
            if self._labels is None:
                raise ValueError(
                    "For list scatter input, provide labels=... (or add boxplots first to "
                    "define labels)."
                )
            labels = self._labels

        if len(vals) != len(labels):
            raise ValueError(
                f"Scatter values length {len(vals)} does not match labels length {len(labels)}."
            )
        return dict(zip(labels, map(float, vals), strict=True))

    def add_scatter_set(
        self,
        values: dict[str, float] | list[float] | pd.Series | pd.DataFrame,
        *,
        name: str | None = None,
        face_color: Color = "black",
        face_alpha: float | None = None,
        marker: str = "o",
        markersize: float = 7.0,
        markeredgecolor: Color = "black",
        markeredgealpha: float | None = None,
        markeredgewidth: float = 0.8,
        labels: list[str] | None = None,
        column: str | None = None,
        x_offset: float | None = None,
        zorder: int = 4,
        add_extra_labels: bool = False,
    ) -> None:
        """Add a set of scatter points to the figure.

        Args:
            values (dict[str, float] | list[float] | pd.Series | pd.DataFrame):
                The scatter values. Can be a dictionary mapping labels to values,
                a list of values, a Series, or a DataFrame.

        Kwargs:
            name (str | None, optional): The name of the scatter set. Defaults to None.
            face_color (Color, optional): The face color of the scatter points. Defaults to "black".
            face_alpha (float | None, optional): The alpha transparency of the scatter points.
                Defaults to None.
            marker (str, optional): The marker style for the scatter points. Defaults to "o".
            markersize (float, optional): The size of the scatter point markers. Defaults to 7.0.
            markeredgecolor (Color, optional): The edge color of the scatter point markers.
                Defaults to "black".
            markeredgealpha (float | None, optional): The alpha transparency of the scatter
                point marker edges. Defaults to None.
            markeredgewidth (float, optional): The width of the scatter point marker edges.
                Defaults to 0.8.
            labels (list[str] | None, optional): The labels for the scatter values
                if values is a list. Defaults to None.
            column (str | None, optional): The column name to use if values is a DataFrame.
            x_offset (float | None, optional): An absolute x-offset from category center.
                Defaults to None.
            zorder (int, optional): The z-order of the scatter points. Defaults to 4.
            add_extra_labels (bool, optional): Whether to allow adding new labels.
                Defaults to False.

        Raises:
            ValueError: If the labels of the incoming scatter set do not match the existing labels
                and add_extra_labels is False.

        Returns:
            None
        """
        values_dict = self._convert_scatter_to_dict(values, labels, column=column)

        incoming = list(values_dict.keys())
        if self._labels is None:
            self._labels = incoming
        else:
            if incoming != self._labels:
                if not add_extra_labels:
                    raise ValueError(
                        "Scatter set labels must match existing labels in the same order.\n"
                        f"Expected: {self._labels}\nGot:      {incoming}\n"
                        "If you want to allow for additional labels, set add_extra_labels=True."
                    )
                self._labels = list(dict.fromkeys(self._labels + incoming))

        set_name = name or f"Scatter {len(self._scatter_data_list) + 1}"
        self._scatter_data_list.append(
            ScatterSetData(
                name=set_name,
                values_dict=values_dict,
                point_data=ScatterPointSettings(
                    markerfacecolor=face_color,
                    markerfacealpha=face_alpha,
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

    def remove_boxplot_group_vlines(self) -> None:
        """Remove vertical lines at the center of boxplot groups."""
        self._include_boxplot_group_vlines = False

    def update_boxplot_group_vline_settings(
        self,
        *,
        linecolor: Color = "#cccccc",
        linealpha: float = 1.0,
        linestyle: str = "-",
        linewidth: float = 0.8,
        zorder: int = -3,
    ) -> None:
        """Update the settings for vertical lines at the center of boxplot groups.

        Kwargs:
            linecolor (Color, optional): The color of the vertical lines. Defaults to "#cccccc".
            linealpha (float, optional): The alpha transparency of the vertical lines.
                Defaults to 1.0.
            linestyle (str, optional): The linestyle of the vertical lines. Defaults to "-".
            linewidth (float, optional): The width of the vertical lines. Defaults to 0.8.
            zorder (int, optional): The z-order of the vertical lines. Defaults to -3.

        Returns:
            None
        """
        self._include_boxplot_group_vlines = True
        self._boxplot_group_vline_settings = LineData(
            value=float("inf"),  # placeholder
            linecolor=linecolor,
            linealpha=linealpha,
            linestyle=linestyle,
            linewidth=linewidth,
            zorder=zorder,
        )

    def clear_vertical_lines_and_bands(self) -> None:
        """Clear all vertical lines and bands from the figure."""
        self._include_boxplot_group_vlines = False
        self._vertical_lines.clear()
        self._vertical_bands.clear()

    def _default_x_tick_locations(self) -> list[float] | None:
        """Get default x-tick locations at the center of each boxplot group."""
        return list(self._boxplot_centers)

    def _default_x_tick_labels(self, tick_locations: list[float]) -> list[str] | None:
        """Get default x-tick labels for the boxplot categories."""
        if self._labels is None:
            return None
        # Only apply category labels when lengths match; if the user overrides locations to
        # something else, leave labels alone unless they explicitly set them.
        if len(tick_locations) == len(self._labels):
            return list(self._labels)
        return None

    def _draw_boxplot_group_vlines(self) -> None:
        """Draw vertical lines at the center of boxplot groups."""
        for x in self._boxplot_centers:
            self.ax.axvline(
                x,
                color=mcolors.to_rgba(
                    self._boxplot_group_vline_settings.linecolor,
                    alpha=self._boxplot_group_vline_settings.linealpha,
                ),
                linestyle=self._boxplot_group_vline_settings.linestyle,
                linewidth=self._boxplot_group_vline_settings.linewidth,
                zorder=self._boxplot_group_vline_settings.zorder,
            )

    @property
    def _boxplot_centers(self) -> np.ndarray:
        """Calculate the x-axis centers for each boxplot category."""
        if self._labels is None:
            return np.array([])

        n_categories = len(self._labels)
        centers = 1.0 + np.arange(n_categories, dtype=float)
        return centers

    def _draw_scatter_sets(self, centers: np.ndarray, *, span: float | None = None) -> None:
        """Draw scatter sets on the plot.

        Args:
            centers (np.ndarray): The x-axis centers for each boxplot category.

        Kwargs:
            span (float | None, optional): The total span for auto-offsetting scatter sets.
                Defaults to None, which uses 80% of boxplot_group_width or 0.8, whichever is
                smaller.

        Returns:
            None
        """
        if len(self._scatter_data_list) == 0 or self._labels is None:
            return

        n = len(self._scatter_data_list)

        # auto-offsets to reduce overlap between multiple scatter sets (still “lined up” per label)
        if span is None:
            span = min(self.boxplot_group_width * 0.8, 0.8)

        auto_offsets = (np.arange(n) - (n - 1) / 2.0) * (span / max(n, 1))

        for i, sdata in enumerate(self._scatter_data_list):
            offset = float(sdata.x_offset) if sdata.x_offset is not None else float(auto_offsets[i])
            xs = centers + offset

            ys = np.array([sdata.values_dict.get(lab, np.nan) for lab in self._labels], dtype=float)
            mask = np.isfinite(ys)
            if not np.any(mask):
                continue
            self.ax.plot(
                xs[mask],
                ys[mask],
                linestyle="none",
                clip_on=True,
                rasterized=True,
                **sdata.point_data.to_mpl_settings_dict(),
            )

    def _build_plot(self) -> Axes:
        """Build the boxplot figure."""
        if self._labels is None or len(self._labels) == 0:
            raise ValueError("No labels defined yet.")

        if len(self._boxplot_data_list) == 0:
            raise ValueError("No boxplot sets added yet.")

        ax = self.ax
        ax.clear()

        n_boxplot_sets = len(self._boxplot_data_list)

        # Category centers on x axis
        centers = self._boxplot_centers

        # Width and offsets for each set within a category
        box_width = self.boxplot_group_width / n_boxplot_sets
        # center the offsets around the category center
        offsets = (np.arange(n_boxplot_sets) - (n_boxplot_sets - 1) / 2.0) * box_width
        widths = box_width * self.boxplot_width_scale

        for k, boxplot_data in enumerate(self._boxplot_data_list):
            pos_k_all = centers + offsets[k]
            data_k: list[list[float]] = []
            pos_k: list[float] = []

            for lab, x in zip(self._labels, pos_k_all, strict=True):
                vals = boxplot_data.scores_dict.get(lab, [])
                if vals is None or len(vals) == 0:
                    continue
                data_k.append(list(vals))
                pos_k.append(float(x))

            if len(data_k) == 0:
                continue

            bp = ax.boxplot(
                data_k,
                positions=pos_k,
                widths=widths,
                patch_artist=True,
                showfliers=boxplot_data.showfliers,
                whis=boxplot_data.percentiles,
                flierprops=boxplot_data.fliersettings.to_mpl_settings_dict(),
            )

            edgecolor = mcolors.to_rgba(boxplot_data.edge_color, alpha=boxplot_data.edge_alpha)
            for patch in bp["boxes"]:
                patch.set_facecolor(
                    mcolors.to_rgba(boxplot_data.face_color, alpha=boxplot_data.alpha)
                )
                patch.set_linewidth(boxplot_data.linewidth)
                patch.set_edgecolor(edgecolor)
                patch.set_zorder(boxplot_data.zorder)

            for key in ("whiskers", "caps", "medians", "means"):
                for artist in bp.get(key, []):
                    artist.set_color(edgecolor)
                    artist.set_linewidth(boxplot_data.linewidth)
                    artist.set_zorder(boxplot_data.zorder)

            for artist in bp.get("fliers", []):
                artist.set_zorder(boxplot_data.zorder)

        self._draw_scatter_sets(centers)
        self._draw_verticals()
        self._draw_horizontals()

        if self._include_boxplot_group_vlines:
            self._draw_boxplot_group_vlines()

        self._set_x_axis()
        self._set_y_axis()

        if self.include_legend:
            ax.legend(
                handles=self._legend_handles,
                loc=self.legend_loc,
                bbox_to_anchor=self.legend_bbox_to_anchor,
            )

        return ax

    def _get_boxplot_legend_handles(self) -> list[Any]:
        """Generate legend handles for boxplot sets.

        Returns:
            list[Any]: A list of legend handles for the boxplot sets.
        """
        handles: list[Any] = []

        for boxplot_data in self._boxplot_data_list:
            handles.append(
                Patch(
                    facecolor=mcolors.to_rgba(boxplot_data.face_color, alpha=boxplot_data.alpha),
                    edgecolor=mcolors.to_rgba(
                        boxplot_data.edge_color, alpha=boxplot_data.edge_alpha
                    ),
                    label=boxplot_data.name,
                )
            )

        return handles

    def _get_scatter_legend_handles(self) -> list[Any]:
        """Generate legend handles for scatter sets.

        Returns:
            list[Any]: A list of legend handles for the scatter sets.
        """
        handles: list[Any] = []

        for sdata in self._scatter_data_list:
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
        """Generated legend handles for boxplot and scatter sets."""
        handles: list[Any] = []

        handles.extend(self._get_boxplot_legend_handles())
        handles.extend(self._get_scatter_legend_handles())
        handles.extend(self._get_named_line_legend_handles())
        handles.extend(self._get_named_band_legend_handles())

        return handles
