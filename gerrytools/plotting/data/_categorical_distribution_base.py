from __future__ import annotations

from numbers import Real
from typing import Mapping, Sequence, cast

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from gerrytools.plotting.data._gerryplot_dataclasses import LineData, PointSetData
from gerrytools.plotting.data.gerryplot import GerryPlotBase
from gerrytools.plotting.mpl.marker_options import PointMarkerOptions
from gerrytools.typing import Color, LegendHandle


class CategoricalDistributionPlotBase(GerryPlotBase):
    """Shared base for categorical distribution plots.

    This class centralizes label alignment, point-set overlays, category-center guides,
    and default x-axis tick behavior used by plots such as ``BoxPlot`` and ``ViolinPlot``.
    """

    def __init__(
        self,
        *,
        figure_size: tuple[float, float] | None,
        dpi: int | None,
        ax: Axes | None,
        include_legend: bool,
        xlabel: str | None,
        ylabel: str | None,
        title: str | None,
        group_width: float,
        width_scale: float,
        include_group_vlines: bool,
    ) -> None:
        """Initialize shared categorical distribution plot state.

        Args:
            figure_size (tuple[float, float]): Figure size in inches.
            dpi (int): Figure resolution in dots per inch.
            include_legend (bool): Whether legend rendering is enabled.
            xlabel (str | None): X-axis label text. Defaults to None.
            ylabel (str | None): Y-axis label text. Defaults to None.
            title (str | None): Plot title text. Defaults to None.
            group_width (float): Width allocated to each category group on the x-axis.
                Must be in ``(0, 1]``.
            width_scale (float): Relative width scaling for grouped artists.
                Must be in ``(0, 1]``.
            include_group_vlines (bool): Whether to show center guide lines at each category.
        """
        super().__init__(
            figure_size=figure_size,
            dpi=dpi,
            ax=ax,
            include_legend=include_legend,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
        )

        if group_width <= 0:
            raise ValueError("group_width must be positive.")
        if group_width > 1.0:
            raise ValueError("group_width must be <= 1.0 when centers are integers.")
        if not 0.0 < width_scale <= 1.0:
            raise ValueError("width_scale must be in (0.0, 1.0].")

        self.group_width = float(group_width)
        self.width_scale = float(width_scale)

        self._pointset_data_list: list[PointSetData] = []
        self._labels: list[str] | None = None

        self._include_group_vlines = include_group_vlines
        self._group_vline_settings = LineData(
            values=float("inf"),  # placeholder
            linecolor="#cccccc",
            linealpha=1.0,
            linestyle="-",
            linewidth=0.8,
            zorder=-3,
        )

    @staticmethod
    def _convert_distribution_data_to_dictionary(
        scores: (
            Mapping[str, Sequence[float]]
            | Sequence[float]
            | Sequence[Sequence[float]]
            | pd.DataFrame
        ),
        scores_labels: list[str] | None = None,
    ) -> dict[str, list[float]]:
        """Convert distribution input to a dictionary mapping labels to score lists.

        Args:
            scores (dict[str, list[float]] | list[float] | list[list[float]] | pd.DataFrame):
                Distribution values. DataFrames use each column as a category.
            scores_labels (list[str] | None, optional): Labels for list-based input.
                Required when ``scores`` is provided as ``list[float]`` or ``list[list[float]]``.
                Defaults to None.

        Returns:
            dict[str, list[float]]: Category label to score-list mapping.
        """
        if isinstance(scores, dict):
            typed_scores = cast(dict[str, Sequence[float]], scores)
            return {str(k): list(v) for k, v in typed_scores.items()}

        if isinstance(scores, pd.DataFrame):
            return {str(col): scores[col].dropna().tolist() for col in scores.columns}

        if isinstance(scores, Sequence):
            if scores_labels is None:
                raise ValueError(
                    "When providing lists of scores, also provide labels for each list."
                )

            if len(scores) == 0:
                raise ValueError("scores is empty; provide at least one score list.")

            first = scores[0]

            def _is_scalar(x: object) -> bool:
                """Return whether an object should be treated as a scalar score value.

                Args:
                    x (object): Candidate object.

                Returns:
                    bool: True when ``x`` is scalar-like for score parsing.
                """
                return isinstance(x, (str, bytes, Real, np.generic))

            def _is_score_series(x: object) -> bool:
                """Return whether an object should be treated as a score sequence.

                Args:
                    x (object): Candidate object.

                Returns:
                    bool: True when ``x`` behaves like a sequence of scores.
                """
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

    def _sync_labels(
        self,
        incoming: list[str],
        *,
        add_extra_labels: bool,
        item_name: str,
    ) -> None:
        """Validate and update plot labels for an incoming dataset.

        Args:
            incoming (list[str]): Label sequence extracted from the incoming dataset.
            add_extra_labels (bool): Whether to extend existing labels with unseen values.
            item_name (str): Dataset descriptor used in validation messages.

        Returns:
            None

        Raises:
            ValueError: If labels differ and ``add_extra_labels`` is False.
        """
        if self._labels is None:
            self._labels = incoming
            return

        if incoming != self._labels:
            if not add_extra_labels:
                raise ValueError(
                    f"{item_name} labels must match existing labels in the same order.\n"
                    f"Expected: {self._labels}\nGot:      {incoming}\n"
                    "If you want to allow additional labels, set add_extra_labels=True."
                )
            self._labels = list(dict.fromkeys(self._labels + incoming))

    def _convert_pointset_to_dict(
        self,
        values: dict[str, float] | list[float] | pd.Series | pd.DataFrame,
        labels: list[str] | None = None,
        *,
        column: str | None = None,
    ) -> dict[str, float]:
        """Convert point-set input to a dictionary mapping labels to float values.

        Args:
            values (dict[str, float] | list[float] | pd.Series | pd.DataFrame): Point values.
            labels (list[str] | None, optional): Labels for list-based input. Defaults to None.
            column (str | None, optional): DataFrame column to read when ``values`` is a
                DataFrame with more than one column. Defaults to None.

        Returns:
            dict[str, float]: Category label to point value mapping.
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
                    "For list pointset input, provide labels=... (or add distributions first to "
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
        """Add a marker point set across categorical x positions.

        Args:
            values (dict[str, float] | list[float] | pd.Series | pd.DataFrame): Point values.
            labels (list[str] | None, optional): Labels for list-based input. Defaults to None.
            column (str | None, optional): DataFrame column to read when ``values`` is a
                DataFrame with more than one column. Defaults to None.
            name (str | None, optional): Legend/display label for this point set.
                Defaults to None.
            facecolor (Color, optional): Marker face color. Defaults to ``"black"``.
            facealpha (float | None, optional): Marker face alpha in ``[0, 1]``. Defaults to None.
            marker (str, optional): Marker style passed to Matplotlib. Defaults to ``"o"``.
            markersize (float, optional): Marker size. Defaults to ``7.0``.
            markeredgecolor (Color, optional): Marker edge color. Defaults to ``"black"``.
            markeredgealpha (float | None, optional): Marker edge alpha in ``[0, 1]``.
                Defaults to None.
            markeredgewidth (float, optional): Marker edge width. Defaults to ``0.8``.
            x_offset (float | None, optional): Explicit horizontal offset from the category
                center. If None, offsets are assigned automatically. Defaults to None.
            zorder (int, optional): Matplotlib z-order. Defaults to ``2``.
            add_extra_labels (bool, optional): If True, allows this point set to introduce
                additional labels beyond the current label set. Defaults to ``False``.
        """
        values_dict = self._convert_pointset_to_dict(values, labels, column=column)
        incoming = list(values_dict.keys())
        self._sync_labels(incoming, add_extra_labels=add_extra_labels, item_name="point set")

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
        self._claim_legend_if_named(name)

    def remove_group_vlines(self) -> None:
        """Disable vertical guide lines at category centers."""
        self._include_group_vlines = False

    def update_group_vline_settings(
        self,
        *,
        linecolor: Color = "#cccccc",
        linealpha: float = 1.0,
        linestyle: str = "-",
        linewidth: float = 0.8,
        zorder: int = -3,
    ) -> None:
        """Update vertical guide-line style for category centers.

        Args:
            linecolor (Color, optional): Guide line color. Defaults to ``"#cccccc"``.
            linealpha (float, optional): Guide line alpha in ``[0, 1]``. Defaults to ``1.0``.
            linestyle (str, optional): Matplotlib line style. Defaults to ``"-"``.
            linewidth (float, optional): Guide line width. Defaults to ``0.8``.
            zorder (int, optional): Matplotlib z-order. Defaults to ``-3``.
        """
        self._include_group_vlines = True
        self._group_vline_settings = LineData(
            values=float("inf"),
            linecolor=linecolor,
            linealpha=linealpha,
            linestyle=linestyle,
            linewidth=linewidth,
            zorder=zorder,
        )

    def clear_verticals(self) -> None:
        """Clear all vertical overlays and disable category-center guide lines."""
        self._include_group_vlines = False
        self._annotations.clear_verticals()

    def _default_x_tick_locations(self) -> list[float] | None:
        """Get default x-tick locations at the center of each category group."""
        return list(self._category_centers)

    def _default_x_tick_labels(self, tick_locations: list[float]) -> list[str] | None:
        """Get default x-tick labels for categories when lengths align.

        Args:
            tick_locations (list[float]): Candidate x-tick positions.

        Returns:
            list[str] | None: Category labels when lengths align; otherwise None.
        """
        if (
            self._labels is None
        ):  # pragma: no cover - _labels is always set before tick helpers are called
            return None  # pragma: no cover
        if len(tick_locations) == len(self._labels):
            return list(self._labels)
        return None

    def _draw_group_vlines(self) -> None:
        """Draw vertical guide lines at each category center."""
        for x in self._category_centers:
            line = self._ax.axvline(
                x,
                color=self._resolved_rgba(
                    self._group_vline_settings.linecolor,
                    self._group_vline_settings.linealpha,
                    field="linecolor",
                ),
                linestyle=self._group_vline_settings.linestyle,
                linewidth=self._group_vline_settings.linewidth,
                zorder=self._group_vline_settings.zorder,
            )
            self._artists.track(line)

    @property
    def _category_centers(self) -> np.ndarray:
        """Calculate x-axis centers for each category."""
        if self._labels is None:
            return np.array([])
        n_categories = len(self._labels)
        return 1.0 + np.arange(n_categories, dtype=float)

    def _draw_pointset(self, centers: np.ndarray, *, span: float | None = None) -> None:
        """Draw all configured point sets on the plot.

        Args:
            centers (np.ndarray): X-axis category centers.
            span (float | None, optional): Total horizontal span used to distribute multiple
                point sets within each category. Defaults to None.

        Returns:
            None
        """
        if len(self._pointset_data_list) == 0 or self._labels is None:
            return

        n = len(self._pointset_data_list)
        if span is None:
            span = min(self.group_width * 0.8, 0.8)

        auto_offsets = (np.arange(n) - (n - 1) / 2.0) * (span / max(n, 1))

        for i, sdata in enumerate(self._pointset_data_list):
            offset = float(sdata.x_offset) if sdata.x_offset is not None else float(auto_offsets[i])
            xs = centers + offset

            ys = np.array([sdata.values_dict.get(lab, np.nan) for lab in self._labels], dtype=float)
            mask = np.isfinite(ys)
            if not np.any(mask):
                continue
            point_lines = self._ax.plot(
                xs[mask],
                ys[mask],
                linestyle="none",
                clip_on=True,
                rasterized=True,
                **sdata.point_data.to_mpl_settings_dict(),
            )
            self._artists.track(point_lines)

    def _get_pointset_legend_handles(self) -> list[LegendHandle]:
        """Generate legend handles for all point sets."""
        handles: list[LegendHandle] = []

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
