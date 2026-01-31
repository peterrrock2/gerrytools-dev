from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from gerrytools.colors import resolve_color_and_alpha
from gerrytools.logging import get_logger
from gerrytools.plotting.gerryplot import (
    GerryPlotBase,
    PointMarkerOptions,
)
from gerrytools.typing import Color

logger = get_logger(__name__)


@dataclass(frozen=True)
class SeaLevelSetData:
    name: str
    scores_dict: dict[str, float]
    linecolor: Color
    linealpha: float | None = None
    linewidth: float = 2.0
    linestyle: str = "-"
    markersettings: PointMarkerOptions = field(default_factory=PointMarkerOptions)
    zorder: int = 1

    def __post_init__(self) -> None:
        lw = float(self.linewidth)
        if not math.isfinite(lw):
            raise ValueError("linewidth must be a finite number")
        if lw < 0:
            raise ValueError("linewidth must be nonnegative")
        object.__setattr__(self, "linewidth", lw)

        resolved_linecolor, resolved_linealpha = resolve_color_and_alpha(
            self.linecolor,
            alpha=self.linealpha,
            allow_none=True,
            field="linecolor",
            owner=f"SeaLevelSetData {self.name}",
            logger=logger,
        )
        object.__setattr__(self, "linecolor", resolved_linecolor)
        object.__setattr__(self, "linealpha", resolved_linealpha)
        object.__setattr__(self, "zorder", int(self.zorder))


class SeaLevel(GerryPlotBase):

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
        """Initialize a SeaLevel instance.

        Args:
            figure_size (tuple[float, float], optional): The size of the figure in inches.
                Defaults to (10, 6).
            dpi (int, optional): The dots per inch (DPI) of the figure. Defaults to 300.
            include_legend (bool, optional): Whether to include a legend in the plot.
                Defaults to True.
        """
        super().__init__(
            figure_size=figure_size,
            dpi=dpi,
            include_legend=include_legend,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
        )

        self.hide_warnings = hide_warnings

        self.grid = grid

        self._sealevel_data_list: list[SeaLevelSetData] = []
        self._labels: list[str] | None = None
        self._maximum_vertical_jitter_per_category: dict[str, float] = {}
        self._maximum_horizontal_jitter_per_category: dict[str, float] = {}

        self._jitter_rng_seed = None
        self._rng = np.random.default_rng(seed=self._jitter_rng_seed)

    @property
    def jitter_rng_seed(self) -> int | None:
        return self._jitter_rng_seed

    @jitter_rng_seed.setter
    def jitter_rng_seed(self, seed: int | None) -> None:
        if seed is not None and not isinstance(seed, int):
            raise TypeError("jitter_rng_seed must be an integer or None.")
        self._jitter_rng_seed = seed
        self._rng = np.random.default_rng(seed=seed)

    def set_max_vertical_jitter_per_category(
        self,
        *,
        jitter_per_category: dict[str, float],
    ) -> None:
        """Set the maximum jitter per category.

        Any points in a given category will be jittered randomly within the range
        [-jitter, +jitter], where jitter is the maximum jitter value for that category

        Any category not specified in the dictionary will have no jitter applied.

        Args:
            jitter_per_category (dict[str, float]): A dictionary mapping category labels
                to maximum jitter values.

        Returns:
            None
        """
        if not isinstance(jitter_per_category, dict):
            raise TypeError("jitter_per_category must be a dictionary mapping labels to floats.")
        if not all(isinstance(v, Real) for v in jitter_per_category.values()):
            raise TypeError("All values in jitter_per_category must be real numbers.")
        if not all(v >= 0 for v in jitter_per_category.values()):
            raise ValueError("All jitter values must be nonnegative.")
        if not all(math.isfinite(float(v)) for v in jitter_per_category.values()):
            raise ValueError("All jitter values must be finite.")
        if len(self._labels or []) == 0:
            raise ValueError("No labels defined yet; cannot set jitter per category.")
        if not set(jitter_per_category.keys()).issubset(set(self._labels or [])):
            extra_keys = set(jitter_per_category.keys()) - set(self._labels or [])
            raise ValueError(
                "All keys in jitter_per_category must be among the existing labels."
                f" Extra keys: {extra_keys}"
            )

        self._maximum_vertical_jitter_per_category = jitter_per_category

    def set_max_vertical_jitter_all(
        self,
        max_jitter: float,
    ) -> None:
        """Set the maximum jitter per category.

        Any points in a given category will be jittered randomly within the range
        [-jitter, +jitter], where jitter is the maximum jitter value for that category

        Any category not specified in the dictionary will have no jitter applied.

        Args:
            jitter_per_category (dict[str, float]): A dictionary mapping category labels
                to maximum jitter values.

        Returns:
            None
        """
        if len(self._labels or []) == 0:
            raise ValueError("No labels defined yet; cannot set jitter per category.")

        max_jit = float(max_jitter)
        if not math.isfinite(max_jit) or max_jit < 0:
            raise ValueError("max_jitter must be a finite nonnegative number.")
        jitter_dict = {label: max_jit for label in self._labels or []}
        self.set_max_vertical_jitter_per_category(jitter_per_category=jitter_dict)

    def set_max_horizontal_jitter_per_category(
        self,
        *,
        jitter_per_category: dict[str, float],
    ) -> None:
        """Set the maximum horizontal jitter per category.

        Any points in a given category will be jittered randomly within the range
        [-jitter, +jitter], where jitter is the maximum jitter value for that category

        Any category not specified in the dictionary will have no jitter applied.

        Args:
            jitter_per_category (dict[str, float]): A dictionary mapping category labels
                to maximum jitter values.

        Returns:
            None
        """
        if not isinstance(jitter_per_category, dict):
            raise TypeError("jitter_per_category must be a dictionary mapping labels to floats.")
        if not all(isinstance(v, Real) for v in jitter_per_category.values()):
            raise TypeError("All values in jitter_per_category must be real numbers.")
        if not all(v >= 0 for v in jitter_per_category.values()):
            raise ValueError("All jitter values must be nonnegative.")
        if not all(math.isfinite(float(v)) for v in jitter_per_category.values()):
            raise ValueError("All jitter values must be finite.")
        if len(self._labels or []) == 0:
            raise ValueError("No labels defined yet; cannot set jitter per category.")
        if not set(jitter_per_category.keys()).issubset(set(self._labels or [])):
            extra_keys = set(jitter_per_category.keys()) - set(self._labels or [])
            raise ValueError(
                "All keys in jitter_per_category must be among the existing labels."
                f" Extra keys: {extra_keys}"
            )

        self._maximum_horizontal_jitter_per_category = jitter_per_category

    def set_max_horizontal_jitter_all(
        self,
        max_jitter: float,
    ) -> None:
        """Set the maximum horizontal jitter per category.

        Any points in a given category will be jittered randomly within the range
        [-jitter, +jitter], where jitter is the maximum jitter value for that category

        Any category not specified in the dictionary will have no jitter applied.

        Args:
            jitter_per_category (dict[str, float]): A dictionary mapping category labels
                to maximum jitter values.

        Returns:
            None
        """
        if len(self._labels or []) == 0:
            raise ValueError("No labels defined yet; cannot set jitter per category.")
        max_jit = float(max_jitter)
        if not math.isfinite(max_jit) or max_jit < 0:
            raise ValueError("max_jitter must be a finite nonnegative number.")
        jitter_dict = {label: max_jit for label in self._labels or []}
        self.set_max_horizontal_jitter_per_category(jitter_per_category=jitter_dict)

    def _convert_score_data_to_dictionary(
        self,
        scores: dict[str, int | float] | list[int | float] | pd.Series | pd.DataFrame,
        scores_labels: list[str] | None = None,
        df_row_index: Any | None = None,
    ) -> dict[str, float]:
        """Convert incoming score data to a dictionary."""
        out_dict: dict[str, float] = {}
        if isinstance(scores, dict):
            out_dict = {str(k): float(v) for k, v in scores.items()}
        elif isinstance(scores, list):
            if scores_labels is None:
                raise ValueError(
                    "If scores is a list, scores_labels must be provided to map values to labels."
                )
            if len(scores) != len(scores_labels):
                raise ValueError("Length of scores list must match length of scores_labels list.")
            out_dict = {str(label): float(value) for label, value in zip(scores_labels, scores)}
        elif isinstance(scores, pd.Series):
            out_dict = {str(idx): float(value) for idx, value in scores.items()}
        elif isinstance(scores, pd.DataFrame):
            if df_row_index is None:
                raise ValueError(
                    "If scores is a DataFrame, df_row_index must be provided to select the row."
                )
            if df_row_index not in scores.index:
                raise ValueError(f"df_row_index {df_row_index} not found in DataFrame index.")
            row = scores.loc[df_row_index]
            if isinstance(row, pd.DataFrame):
                raise ValueError(
                    "The selected df_row_index corresponds to multiple rows. "
                    "Please provide a df_row_index that selects a single row."
                )
            out_dict = {str(idx): float(value) for idx, value in row.items()}

        if out_dict:
            if any(not math.isfinite(v) for v in out_dict.values()):
                raise ValueError("All score values must be finite numbers.")
            return out_dict

        if isinstance(scores, (dict, list, pd.Series, pd.DataFrame)):
            raise ValueError(
                "Could not convert scores to dictionary. Please check that the "
                "input is not empty."
            )

        raise TypeError(
            "Could not convert scores to dictionary. "
            "Scores must be a dict, list, pd.Series, or pd.DataFrame. "
            "If a list is provided, scores_labels must also be provided. "
            "If a DataFrame is provided, df_row_index must also be provided."
        )

    def add_sealevel_set(
        self,
        scores: dict[str, float] | list[float] | pd.Series | pd.DataFrame,
        *,
        scores_labels: list[str] | None = None,
        df_row_index: Any | None = None,
        name: str | None = None,
        linecolor: Color = "black",
        linealpha: float | None = None,
        linewidth: float = 1.5,
        linestyle: str = "-",
        marker: str = "o",
        markerfacecolor: Color | None = None,
        markerfacealpha: float | None = None,
        markersize: float = 7.0,
        markeredgecolor: Color | None = None,
        markeredgealpha: float | None = None,
        markeredgewidth: float = 0.8,
        zorder: int = 2,
    ) -> None:
        """Add a set of points to the figure.

        Args:
            scores (dict[str, float] | list[float] | pd.Series | pd.DataFrame):
                The pointset values. Can be a dictionary mapping labels to values,
                a list of values, a Series, or a DataFrame.
            scores_labels (list[str] | None, optional): The labels corresponding to the
                scores list, if scores is provided as a list. Ignored if scores is a dict,
                Series, or DataFrame. Defaults to None.
            df_row_index (Any | None, optional): The row index to select if scores is a DataFrame.
                Defaults to None.
            name (str | None, optional): The name of the point set for the legend.
                Defaults to None.
            linecolor (Color, optional): The color of the line connecting the points.
                Defaults to "black".
            linealpha (float | None, optional): The alpha transparency of the line.
                Defaults to None.
            linewidth (float, optional): The width of the line. Defaults to 1.5.
            linestyle (str, optional): The style of the line. Defaults to "-".
            marker (str, optional): The marker style for the points. Defaults to "o".
            markerfacecolor (Color | None, optional): The face color of the markers.
                Defaults to None, which uses linecolor.
            markerfacealpha (float | None, optional): The alpha transparency of the marker face.
                Defaults to None, which uses linealpha.
            markersize (float, optional): The size of the markers. Defaults to 7.0.
            markeredgecolor (Color | None, optional): The edge color of the markers.
                Defaults to None, which uses markerfacecolor if defined and linecolor otherwise.
            markeredgealpha (float | None, optional): The alpha transparency of the marker edge.
                Defaults to None, which uses markerfacealpha if defined and linealpha otherwise.
            markeredgewidth (float, optional): The width of the marker edge. Defaults to 0.8.
            zorder (int, optional): The z-order for layering the plot elements.
                Defaults to 2.

        Returns:
            None
        """
        scores_dict = self._convert_score_data_to_dictionary(scores, scores_labels, df_row_index)

        if markerfacecolor is None:
            markerfacecolor = linecolor
        if markerfacealpha is None:
            markerfacealpha = linealpha
        if markeredgecolor is None:
            markeredgecolor = markerfacecolor if markerfacecolor is not None else linecolor
        if markeredgealpha is None:
            markeredgealpha = markerfacealpha if markerfacealpha is not None else linealpha

        if self._labels is None:
            self._labels = list(scores_dict.keys())
        else:
            incoming = list(scores_dict.keys())
            if incoming != self._labels:
                raise ValueError(
                    "All sets must use the same labels in the same order.\n"
                    f"Expected: {self._labels}\nGot:      {incoming}\n"
                    "If you want to allow for additional labels, set add_extra_labels=True."
                )

        set_name = name or f"Set {len(self._sealevel_data_list) + 1}"
        self._sealevel_data_list.append(
            SeaLevelSetData(
                name=set_name,
                scores_dict=scores_dict,
                linecolor=linecolor,
                linealpha=linealpha,
                linewidth=linewidth,
                linestyle=linestyle,
                markersettings=PointMarkerOptions(
                    markerfacecolor=markerfacecolor,
                    markerfacealpha=markerfacealpha,
                    marker=marker,
                    markersize=markersize,
                    markeredgecolor=markeredgecolor,
                    markeredgealpha=markeredgealpha,
                    markeredgewidth=markeredgewidth,
                ),
                zorder=zorder,
            )
        )

    @property
    def _sealevel_centers(self) -> np.ndarray:
        """Calculate the x-axis centers for each sealevel category."""
        if self._labels is None:
            return np.array([])

        n_categories = len(self._labels)
        centers = 1.0 + np.arange(n_categories, dtype=float)
        return centers

    def _default_x_tick_locations(self) -> list[float] | None:
        """Get default x-tick locations at the center of each sealevel group."""
        return list(self._sealevel_centers)

    def _default_x_tick_labels(self, tick_locations: list[float]) -> list[str] | None:
        """Get default x-tick labels for the sealevel categories."""
        if self._labels is None:
            return None
        # Only apply category labels when lengths match; if the user overrides locations to
        # something else, leave labels alone unless they explicitly set them.
        if len(tick_locations) == len(self._labels):
            return list(self._labels)
        return None

    def _draw_sealevels(self) -> None:
        """Draw the sealevel sets on the plot."""
        centers = self._sealevel_centers

        if self._labels is None:
            raise ValueError("No labels defined yet.")

        for sealevel_set in self._sealevel_data_list:
            x_positions = []
            y_positions = []
            for idx, label in enumerate(self._labels or []):
                hoizontal_jitter = self._maximum_horizontal_jitter_per_category.get(label, 0.0)
                x_center = centers[idx] + self._rng.uniform(
                    low=-hoizontal_jitter,
                    high=hoizontal_jitter,
                )
                x_positions.append(x_center)

                vertical_jitter = self._maximum_vertical_jitter_per_category.get(label, 0.0)
                y_center = sealevel_set.scores_dict[label] + self._rng.uniform(
                    low=-vertical_jitter,
                    high=vertical_jitter,
                )
                y_positions.append(y_center)

            markersettings = sealevel_set.markersettings.to_mpl_settings_dict()
            markersettings.pop("zorder", None)
            self._ax.plot(
                x_positions,
                y_positions,
                linestyle=sealevel_set.linestyle,
                color=mcolors.to_rgba(sealevel_set.linecolor, sealevel_set.linealpha),
                linewidth=sealevel_set.linewidth,
                zorder=sealevel_set.zorder,
                label=sealevel_set.name,
                **markersettings,
            )

    def _build_plot(self) -> None:
        """Build the sealevel figure."""
        if self._labels is None or len(self._labels) == 0:
            raise ValueError("No labels defined yet.")

        if len(self._sealevel_data_list) == 0:
            raise ValueError("No sealevel sets added yet.")

        self._draw_sealevels()

    def _get_sealevel_legend_handles(self) -> list[Any]:
        """Generate legend handles for sealevel sets.

        Returns:
            list[Any]: A list of legend handles for the sealevel sets.
        """
        handles: list[Any] = []

        for sealevel_data in self._sealevel_data_list:
            markersettings = sealevel_data.markersettings.to_mpl_settings_dict()
            markersettings.pop("zorder", None)
            handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle=sealevel_data.linestyle,
                    color=mcolors.to_rgba(sealevel_data.linecolor, sealevel_data.linealpha),
                    linewidth=sealevel_data.linewidth,
                    label=sealevel_data.name,
                    zorder=sealevel_data.zorder,
                    **markersettings,
                )
            )

        return handles

    def format_ylabels_as_fractions(
        self, denominator: int, *, minimum_numerator: int = 0, maximum_numerator: int | None = None
    ) -> None:
        """Format y-axis labels as fractions out of the given denominator.

        Args:
            denominator (int): The denominator to use for the fractions.
            minimum_numerator (int, optional): The minimum numerator to display. Defaults to 0.
            maximum_numerator (int | None, optional): The maximum numerator to display.
                Defaults to None, which uses the denominator as the maximum numerator.

        Returns:
            None
        """
        if not isinstance(denominator, int):
            raise TypeError("denominator must be an integer.")
        if denominator <= 0:
            raise ValueError("denominator must be a positive integer.")

        original_maximum = maximum_numerator
        if maximum_numerator is None:
            maximum_numerator = denominator
        if not isinstance(minimum_numerator, int):
            raise TypeError("minimum_numerator must be an integer.")
        if not isinstance(maximum_numerator, int):
            raise TypeError("maximum_numerator must be an integer.")
        if maximum_numerator > denominator:
            raise ValueError("maximum_numerator cannot exceed denominator.")
        if minimum_numerator > maximum_numerator:
            if original_maximum is None:
                raise ValueError(
                    "minimum_numerator cannot exceed denominator when maximum_numerator is not set."
                )
            raise ValueError("minimum_numerator cannot exceed maximum_numerator.")

        self.update_ytick_values(
            locations=[n / denominator for n in range(minimum_numerator, maximum_numerator + 1)],
            labels=[f"{n}/{denominator}" for n in range(minimum_numerator, maximum_numerator + 1)],
        )

    @property
    def _legend_handles(self) -> list[Any]:
        """Generated legend handles for sealevel sets."""
        handles: list[Any] = []
        handles.extend(self._get_sealevel_legend_handles())
        handles.extend(self._get_named_line_legend_handles())
        handles.extend(self._get_named_band_legend_handles())

        return handles
