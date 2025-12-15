from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from gerrytools.logging import get_logger
from gerrytools.plotting.colors import HEX8_OR_NONE_PATTERN, convert_color_to_hexa_or_none
from gerrytools.plotting.gerryplot import (
    GerryPlotBase,
    LineData,
    ScatterPointSettings,
)
from gerrytools.typing import Color

logger = get_logger(__name__)


@dataclass(frozen=True)
class ScatterSetData:
    name: str
    values_dict: dict[str, float]  # one value per label
    point_data: ScatterPointSettings
    x_offset: float | None = None  # optional absolute x-offset from category center


@dataclass(frozen=True)
class BoxPlotSetData:
    name: str
    scores_dict: dict[str, list[float]]
    face_color: Color
    alpha: float | None = None
    percentiles: tuple[float, float] = (1, 99)
    showfliers: bool = False
    flierdata: ScatterPointSettings = field(default_factory=ScatterPointSettings)
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

        new_color = convert_color_to_hexa_or_none(self.face_color)
        if HEX8_OR_NONE_PATTERN.match(new_color) is None:
            raise ValueError("Invalid color after conversion")

        if new_color.lower() == "none":
            object.__setattr__(self, "face_color", "none")
            object.__setattr__(self, "alpha", 0.0)

        else:
            hex_color, alpha = new_color[:7], int(new_color[7:], 16) / 255.0
            object.__setattr__(self, "face_color", hex_color)

            old_alpha = float(self.alpha) if self.alpha is not None else None
            if old_alpha is not None and not (0.0 <= old_alpha <= 1.0):
                raise ValueError("Alpha must be between 0.0 and 1.0")

            if old_alpha is not None and old_alpha != alpha:
                logger.log(
                    level=logging.DEBUG,
                    msg=(
                        f"For BoxPlotSetData {self.name}: Ignoring alpha from color {new_color} "
                        f"because explicit alpha {self.alpha} was provided."
                    ),
                )
            object.__setattr__(self, "alpha", alpha if old_alpha is None else old_alpha)

        lw = float(self.linewidth)
        if not math.isfinite(lw):
            raise ValueError("linewidth must be a finite number")
        if lw < 0:
            raise ValueError("linewidth must be nonnegative")
        object.__setattr__(self, "linewidth", lw)

        ec_hex8 = convert_color_to_hexa_or_none(self.edge_color)

        if HEX8_OR_NONE_PATTERN.match(ec_hex8) is None:
            raise ValueError("Invalid edge color after conversion")

        if ec_hex8.lower() == "none":
            object.__setattr__(self, "edge_color", "none")
            object.__setattr__(self, "linewidth", 0.0)
            object.__setattr__(self, "edge_alpha", 0.0)
        else:
            object.__setattr__(self, "edge_color", ec_hex8[:7])
            edge_alpha_from_color = int(ec_hex8[7:], 16) / 255.0
            if self.edge_alpha is not None:
                old_edge_alpha = float(self.edge_alpha)
                if not (0.0 <= old_edge_alpha <= 1.0):
                    raise ValueError("edge_alpha must be between 0.0 and 1.0")
                if old_edge_alpha != edge_alpha_from_color:
                    logger.log(
                        level=logging.DEBUG,
                        msg=(
                            f"For BoxPlotSetData {self.name}: Ignoring alpha from edge_color "
                            f"{ec_hex8} because explicit edge_alpha {old_edge_alpha} was provided."
                        ),
                    )
                object.__setattr__(self, "edge_alpha", old_edge_alpha)
            else:
                object.__setattr__(self, "edge_alpha", edge_alpha_from_color)

        object.__setattr__(self, "zorder", int(self.zorder))


class BoxPlot(GerryPlotBase):
    """A class for creating boxplot comparison figures with multiple boxplot sets and scatter sets
    representing a distribution of scores across multiple categories.

    Attributes:
        fig (plt.Figure): The Matplotlib figure object.
        ax (plt.Axes): The Matplotlib axes object.
        boxplot_group_width (float): The width of each boxplot group.
        boxplot_group_gap (float): The gap between boxplot groups.
        include_legend (bool): Whether to include a legend in the plot.
        legend_loc (str): The location of the legend.
        legend_bbox_to_anchor (tuple[float, float]): The bounding box anchor for the legend.


    Methods:
        add_boxplot_set: Add a set of boxplots to the figure.
        add_scatter_set: Add a set of scatter points to the figure.
        add_vertical_line: Add a vertical line to the figure.
        add_vertical_band: Add a vertical band to the figure.
        add_horizontal_line: Add a horizontal line to the figure.
        add_horizontal_band: Add a horizontal band to the figure.
        set_xlimits: Set the x-axis limits.
        set_ylimits: Set the y-axis limits.
        hide_frame: Hide the frame of the plot.
    """

    def __init__(
        self,
        figure_size: tuple[float, float] = (10, 6),
        dpi: int = 300,
        *,
        boxplot_group_width: float = 1.0,
        boxplot_group_gap: float = 0.4,
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
                Defaults to 1.0.
            boxplot_group_gap (float, optional): The gap between boxplot groups.
                Defaults to 0.4.
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
        if boxplot_group_gap < 0:
            raise ValueError("boxplot_group_gap must be nonnegative")
        if not 0 <= boxplot_group_gap <= 0.9:
            raise ValueError("boxplot_group_gap must be between 0.0 and 0.9)")

        self.boxplot_group_width = float(boxplot_group_width)
        self.boxplot_group_gap = float(boxplot_group_gap)

        self._include_boxplot_group_vlines = include_boxplot_group_vlines
        self._boxplot_group_vline_settings = LineData(
            value=float("inf"),  # placeholder
            linecolor="#cccccc",
            linealpha=1.0,
            linestyle="-",
            linewidth=0.8,
            zorder=-3,
        )

    def _convert_boxplot_data_to_dictionary(
        self,
        scores: dict[str, list[float]] | list[float] | list[list[float]] | pd.DataFrame,
        scores_labels: list[str] | None = None,
    ) -> dict[str, list[float]]:
        """Convert boxplot input to a dictionary mapping labels to score lists."""
        if isinstance(scores, dict):
            return {str(k): list(v) for k, v in scores.items()}

        if isinstance(scores, pd.DataFrame):
            return {str(col): scores[col].dropna().tolist() for col in scores.columns}

        if isinstance(scores, list):
            if scores_labels is None:
                raise ValueError(
                    "When providing lists of scores, please also provide labels for each list."
                )

            if len(scores) == 0 or not isinstance(scores[0], list):
                scores_list_of_lists: list[list[float]] = [scores]  # type: ignore[list-item]
            else:
                scores_list_of_lists = scores  # type: ignore[assignment]

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
        face_color: str = "denim",
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
            face_color (str, optional): The face color of the boxplots. Defaults to "denim".
            scores_labels (list[str] | None, optional): The labels for the scores if
                scores is a list or list of lists. Defaults to None.
            percentiles (tuple[float, float], optional): The percentiles for whiskers.
                Defaults to (1, 99).
            showfliers (bool, optional): Whether to show outliers. Defaults to False.
            fliersettings (ScatterPointSettings, optional): The settings for outlier points.
                Defaults to ScatterPointSettings().
            alpha (float | None, optional): The alpha transparency for the boxplots.
                Defaults to None.
            edge_color (Color, optional): The edge color of the boxplots. Defaults to "black".
            add_extra_labels (bool, optional): Whether to allow adding new labels
                not in existing sets. Defaults to False.

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
                flierdata=fliersettings,
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
        """Convert scatter input to a dictionary mapping labels to float values."""
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
        face_color: str = "black",
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
            face_color (str, optional): The color of the scatter points. Defaults to "black".
            face_alpha (float | None, optional): The alpha transparency of the scatter points.
                Defaults to None.
            marker (str, optional): The marker style of the scatter points. Defaults to "o".
            markersize (float, optional): The size of the scatter points. Defaults to 7.0.
            markeredgecolor (str, optional): The edge color of the scatter points.
                Defaults to "black".
            markeredgealpha (float | None, optional): The edge alpha transparency of the
                scatter points. Defaults to None.
            markeredgewidth (float, optional): The edge linewidth of the scatter points.
                Defaults to 0.8.
            labels (list[str] | None, optional): The labels for the scatter values
                if values is a list. Defaults to None.
            column (str | None, optional): The column name to use if values is a DataFrame.
                Defaults to None.
            x_offset (float | None, optional): An optional absolute x-offset from category center.
                Defaults to None.
            zorder (int, optional): The z-order of the scatter points. Defaults to 4.
            add_extra_labels (bool, optional): Whether to allow adding new labels
                not in existing sets. Defaults to False.

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

    def _set_x_axis(self) -> None:
        x_limits = self._x_limits if self._x_limits is not None else self.ax.get_xlim()
        self.ax.set_xlim(x_limits)

        if self._x_tick_locations is not None:
            x_tick_locations = list(self._x_tick_locations)
        else:
            x_tick_locations = list(self._boxplot_centers)

        self.ax.set_xticks(ticks=x_tick_locations)

        if self._x_tick_labels == []:
            self.ax.tick_params(axis="x", labelbottom=False)
            return

        if self._x_tick_labels is None:
            # User explicitly overrode locations (including []): leave existing labels alone.
            if self._x_tick_locations is not None:
                return
            # Default behavior: label categories.
            x_tick_labels = self._labels
        else:
            x_tick_labels = self._x_tick_labels

        if x_tick_labels is None:
            return

        if len(x_tick_labels) != len(x_tick_locations):
            raise ValueError(
                f"Expected {len(x_tick_locations)} x tick labels, got {len(x_tick_labels)}."
            )

        self.ax.set_xticklabels(list(x_tick_labels))

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
        step = self.boxplot_group_width + self.boxplot_group_gap
        centers = 1.0 + np.arange(n_categories) * step
        return centers

    def _draw_scatter_sets(self, centers: np.ndarray, *, span: float | None = None) -> None:
        """Draw scatter sets on the plot."""
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
                linestyle="",
                clip_on=True,
                rasterized=True,
                **sdata.point_data.to_mpl_settings_dict(),
            )

    def _build_plot(self) -> Axes:
        """Build the boxplot figure."""
        if self._labels is None or len(self._labels) == 0:
            raise ValueError("No boxplot sets added yet.")

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
        widths = box_width * 0.9

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
                flierprops=boxplot_data.flierdata.to_mpl_settings_dict(),
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
        """Generate legend handles for boxplot sets."""
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
        """Generate legend handles for scatter sets."""
        handles: list[Any] = []

        for sdata in self._scatter_data_list:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="",
                    label=sdata.name,
                    **sdata.point_data.to_mpl_settings_dict(),
                )
            )

        return handles

    @property
    def _legend_handles(self) -> list[Any]:
        """Generate legend handles for boxplot and scatter sets."""
        handles: list[Any] = []

        handles.extend(self._get_boxplot_legend_handles())
        handles.extend(self._get_scatter_legend_handles())
        handles.extend(self._get_named_line_legend_handles())
        handles.extend(self._get_named_band_legend_handles())

        return handles
