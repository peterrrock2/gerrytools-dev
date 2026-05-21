from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from numpy.random import Generator

from gerrytools.colors import resolve_color_and_alpha
from gerrytools.logging import get_logger
from gerrytools.plotting._rng import resolve_numpy_rng
from gerrytools.plotting.data.gerryplot import GerryPlotBase
from gerrytools.plotting.data.options import SeaLevelLineOptions
from gerrytools.plotting.mpl.marker_options import PointMarkerOptions
from gerrytools.typing import CategoryKey, Color, LegendHandle

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
        figure_size: tuple[float, float] | None = None,
        dpi: int | None = None,
        *,
        ax: Axes | None = None,
        include_legend: bool = True,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        jitter_rng_seed: int | None = None,
        jitter_rng: Generator | None = None,
    ) -> None:
        """Initialize a SeaLevel instance.

        Args:
            figure_size (tuple[float, float] | None, optional): The size of the
                figure in inches. Defaults to ``(10, 6)`` when ``ax`` is not provided.
            dpi (int | None, optional): The dots per inch (DPI) of the figure.
                Defaults to ``300`` when ``ax`` is not provided.
            ax (matplotlib.axes.Axes | None, optional): Render onto an existing
                matplotlib ``Axes`` instead of creating a fresh figure. Defaults to None.
            include_legend (bool, optional): Whether to include a legend in the plot.
                Defaults to True.
            jitter_rng_seed (int | None, optional): Seed for reproducible jitter placement.
                Defaults to None.
            jitter_rng (Generator | None, optional): Explicit NumPy generator to use for
                jitter instead of constructing one from ``jitter_rng_seed``. Defaults to None.

        To toggle the grid or suppress warnings, call :meth:`enable_grid` /
        :meth:`disable_grid` and :meth:`suppress_warnings` / :meth:`show_warnings`
        after construction.
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

        self.hide_warnings = False
        self.grid = False

        self._sealevel_data_list: list[SeaLevelSetData] = []
        self._labels: list[str] | None = None
        self._maximum_vertical_jitter_per_category: dict[str, float] = {}
        self._maximum_horizontal_jitter_per_category: dict[str, float] = {}

        self._jitter_rng, self._jitter_rng_seed = resolve_numpy_rng(
            seed=jitter_rng_seed,
            rng=jitter_rng,
            field_name="jitter_rng_seed",
        )

    def enable_grid(self) -> None:
        """Show a matplotlib grid on the plot."""
        self.grid = True

    def disable_grid(self) -> None:
        """Hide the matplotlib grid (the default)."""
        self.grid = False

    def suppress_warnings(self) -> None:
        """Suppress warnings about potentially problematic configuration."""
        self.hide_warnings = True

    def show_warnings(self) -> None:
        """Re-enable warnings about potentially problematic configuration."""
        self.hide_warnings = False

    @property
    def jitter_rng_seed(self) -> int | None:
        """Get the RNG seed used for deterministic jitter placement.

        Returns:
            int | None: Current jitter RNG seed, or None for nondeterministic behavior.
        """
        return self._jitter_rng_seed

    @jitter_rng_seed.setter
    def jitter_rng_seed(self, seed: int | None) -> None:
        """Set the RNG seed used for deterministic jitter placement.

        Args:
            seed (int | None): Integer seed value, or None to use nondeterministic randomness.

        Returns:
            None

        Raises:
            TypeError: If ``seed`` is neither ``int`` nor ``None``.
        """
        self._jitter_rng, self._jitter_rng_seed = resolve_numpy_rng(
            seed=seed,
            field_name="jitter_rng_seed",
        )

    def _validate_jitter_per_category(self, jitter: dict[str, float]) -> None:
        """Validate a per-category jitter dictionary against the current labels."""
        if not isinstance(jitter, dict):
            raise TypeError("jitter must be a dictionary mapping labels to floats.")
        if not all(isinstance(v, Real) for v in jitter.values()):
            raise TypeError("All values in jitter must be real numbers.")
        if not all(v >= 0 for v in jitter.values()):
            raise ValueError("All jitter values must be nonnegative.")
        if not all(math.isfinite(float(v)) for v in jitter.values()):
            raise ValueError("All jitter values must be finite.")
        if len(self._labels or []) == 0:
            raise ValueError("No labels defined yet; cannot set jitter per category.")
        if not set(jitter.keys()).issubset(set(self._labels or [])):
            extra_keys = set(jitter.keys()) - set(self._labels or [])
            raise ValueError(
                f"All keys in jitter must be among the existing labels. Extra keys: {extra_keys}"
            )

    def _coerce_jitter(self, jitter: float | dict[str, float]) -> dict[str, float]:
        """Resolve a uniform-or-per-category jitter input into a per-category dict.

        - ``dict[str, float]`` is validated and returned as-is.
        - A scalar float is broadcast to every existing label.
        """
        if isinstance(jitter, dict):
            self._validate_jitter_per_category(jitter)
            return jitter
        if not isinstance(jitter, Real):
            raise TypeError(
                "jitter must be a float or a dictionary mapping labels to floats; "
                f"got {type(jitter).__name__!r}."
            )
        if len(self._labels or []) == 0:
            raise ValueError("No labels defined yet; cannot set jitter.")
        jit = float(jitter)
        if not math.isfinite(jit) or jit < 0:
            raise ValueError("jitter must be a finite nonnegative number.")
        return {label: jit for label in self._labels or []}

    def set_vertical_jitter(self, jitter: float | dict[str, float]) -> None:
        """Set the maximum vertical jitter applied to category points.

        Pass a single ``float`` to apply the same jitter uniformly to every
        category, or pass a ``dict`` mapping label to float for per-category
        jitter. Categories not in the dict get no jitter.

        Any point in a given category is jittered randomly within
        ``[-jitter, +jitter]``.

        Args:
            jitter (float | dict[str, float]): Uniform jitter value, or
                per-category mapping.
        """
        self._maximum_vertical_jitter_per_category = self._coerce_jitter(jitter)

    def set_horizontal_jitter(self, jitter: float | dict[str, float]) -> None:
        """Set the maximum horizontal jitter applied to category points.

        Pass a single ``float`` to apply the same jitter uniformly to every
        category, or pass a ``dict`` mapping label to float for per-category
        jitter. Categories not in the dict get no jitter.

        Args:
            jitter (float | dict[str, float]): Uniform jitter value, or
                per-category mapping.
        """
        self._maximum_horizontal_jitter_per_category = self._coerce_jitter(jitter)

    def _convert_score_data_to_dictionary(
        self,
        scores: dict[str, int | float] | list[int | float] | pd.Series | pd.DataFrame,
        scores_labels: list[str] | None = None,
        df_row_index: CategoryKey | None = None,
    ) -> dict[str, float]:
        """Convert supported score inputs into a label-to-value dictionary.

        Args:
            scores (dict[str, int | float] | list[int | float] | pd.Series | pd.DataFrame):
                Input scores. Lists require ``scores_labels``; DataFrames require ``df_row_index``.
            scores_labels (list[str] | None, optional): Labels for list input. Defaults to None.
            df_row_index (CategoryKey | None, optional): Row selector for DataFrame input.
                Defaults to None.

        Returns:
            dict[str, float]: Mapping from category label to numeric value.

        Raises:
            ValueError: If conversion fails, inputs are empty, or values are non-finite.
            TypeError: If ``scores`` uses an unsupported input type.
        """
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
                "Could not convert scores to dictionary. Please check that the input is not empty."
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
        name: str | None = None,
        *,
        scores_labels: list[str] | None = None,
        df_row_index: CategoryKey | None = None,
        line_options: SeaLevelLineOptions | None = None,
        marker_options: PointMarkerOptions | None = None,
        linecolor: Color | None = None,
        linealpha: float | None = None,
        linewidth: float | None = None,
        linestyle: str | None = None,
        marker: str | None = None,
        markerfacecolor: Color | None = None,
        markerfacealpha: float | None = None,
        markersize: float | None = None,
        markeredgecolor: Color | None = None,
        markeredgealpha: float | None = None,
        markeredgewidth: float | None = None,
        zorder: int | None = None,
    ) -> None:
        """Add a set of points to the figure.

        Args:
            scores (dict[str, float] | list[float] | pd.Series | pd.DataFrame):
                The pointset values. Can be a dictionary mapping labels to values,
                a list of values, a Series, or a DataFrame.
            scores_labels (list[str] | None, optional): The labels corresponding to the
                scores list, if scores is provided as a list. Ignored if scores is a dict,
                Series, or DataFrame. Defaults to None.
            df_row_index (CategoryKey | None, optional): The row index to select if scores is a
                DataFrame.
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

        # Resolve line styling: kwargs override line_options, which falls back to defaults.
        line_base = line_options if line_options is not None else SeaLevelLineOptions()
        resolved_linecolor = linecolor if linecolor is not None else line_base.linecolor
        resolved_linealpha = linealpha if linealpha is not None else line_base.linealpha
        resolved_linewidth = linewidth if linewidth is not None else line_base.linewidth
        resolved_linestyle = linestyle if linestyle is not None else line_base.linestyle
        resolved_zorder = zorder if zorder is not None else line_base.zorder

        # Resolve marker styling: kwargs override marker_options, which falls back to a
        # set of defaults that mimic the previous "marker inherits from line" semantics
        # (face=linecolor, edge=face, etc.) when neither is explicitly provided.
        marker_base = (
            marker_options
            if marker_options is not None
            else PointMarkerOptions(
                markerfacecolor=resolved_linecolor,
                markerfacealpha=resolved_linealpha,
                marker="o",
                markersize=7.0,
                markeredgecolor=resolved_linecolor,
                markeredgealpha=resolved_linealpha,
                markeredgewidth=0.8,
            )
        )
        resolved_markerfacecolor = (
            markerfacecolor if markerfacecolor is not None else marker_base.markerfacecolor
        )
        resolved_markerfacealpha = (
            markerfacealpha if markerfacealpha is not None else marker_base.markerfacealpha
        )
        resolved_marker = marker if marker is not None else marker_base.marker
        resolved_markersize = markersize if markersize is not None else marker_base.markersize
        resolved_markeredgecolor = (
            markeredgecolor
            if markeredgecolor is not None
            else (
                marker_base.markeredgecolor
                if marker_base.markeredgecolor != "black"  # PointMarkerOptions default
                else resolved_markerfacecolor
            )
        )
        resolved_markeredgealpha = (
            markeredgealpha if markeredgealpha is not None else marker_base.markeredgealpha
        )
        resolved_markeredgewidth = (
            markeredgewidth if markeredgewidth is not None else marker_base.markeredgewidth
        )

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
                linecolor=resolved_linecolor,
                linealpha=resolved_linealpha,
                linewidth=resolved_linewidth,
                linestyle=resolved_linestyle,
                markersettings=PointMarkerOptions(
                    markerfacecolor=resolved_markerfacecolor,
                    markerfacealpha=resolved_markerfacealpha,
                    marker=resolved_marker,
                    markersize=resolved_markersize,
                    markeredgecolor=resolved_markeredgecolor,
                    markeredgealpha=resolved_markeredgealpha,
                    markeredgewidth=resolved_markeredgewidth,
                ),
                zorder=resolved_zorder,
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
        """Get default x-tick labels for sealevel categories.

        Args:
            tick_locations (list[float]): Candidate x-tick positions.

        Returns:
            list[str] | None: Category labels when the lengths match; otherwise ``None``.
        """
        assert self._labels is not None, (
            "Internal error: _labels should be set before _default_x_tick_labels is called."
        )
        # Only apply category labels when lengths match; if the user overrides locations to
        # something else, leave labels alone unless they explicitly set them.
        if len(tick_locations) == len(self._labels):
            return list(self._labels)
        return None

    def _draw_sealevels(self) -> None:
        """Draw the sealevel sets on the plot."""
        centers = self._sealevel_centers

        assert self._labels is not None, (
            "Internal error: _labels should be set before _draw_sealevels is called."
        )
        for sealevel_set in self._sealevel_data_list:
            x_positions = []
            y_positions = []
            for idx, label in enumerate(self._labels or []):
                hoizontal_jitter = self._maximum_horizontal_jitter_per_category.get(label, 0.0)
                x_center = centers[idx] + self._jitter_rng.uniform(
                    low=-hoizontal_jitter,
                    high=hoizontal_jitter,
                )
                x_positions.append(x_center)

                vertical_jitter = self._maximum_vertical_jitter_per_category.get(label, 0.0)
                y_center = sealevel_set.scores_dict[label] + self._jitter_rng.uniform(
                    low=-vertical_jitter,
                    high=vertical_jitter,
                )
                y_positions.append(y_center)

            markersettings = sealevel_set.markersettings.to_mpl_settings_dict()
            self._ax.plot(
                x_positions,
                y_positions,
                linestyle=sealevel_set.linestyle,
                color=self._resolved_rgba(
                    sealevel_set.linecolor,
                    sealevel_set.linealpha,
                    field="linecolor",
                ),
                linewidth=sealevel_set.linewidth,
                zorder=sealevel_set.zorder,
                label=sealevel_set.name,
                markerfacecolor=markersettings["markerfacecolor"],
                marker=markersettings["marker"],
                markersize=markersettings["markersize"],
                markeredgecolor=markersettings["markeredgecolor"],
                markeredgewidth=markersettings["markeredgewidth"],
            )

    def _build_plot(self) -> None:
        """Build the sealevel figure."""
        if self._labels is None or len(self._labels) == 0:
            raise ValueError("No labels defined yet.")

        if len(self._sealevel_data_list) == 0:
            raise ValueError("No sealevel sets added yet.")

        self._draw_sealevels()

    def _get_sealevel_legend_handles(self) -> list[LegendHandle]:
        """Generate legend handles for sealevel sets.

        Returns:
            list[LegendHandle]: A list of legend handles for the sealevel sets.
        """
        handles: list[LegendHandle] = []

        for sealevel_data in self._sealevel_data_list:
            markersettings = sealevel_data.markersettings.to_mpl_settings_dict()
            handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle=sealevel_data.linestyle,
                    color=self._resolved_rgba(
                        sealevel_data.linecolor,
                        sealevel_data.linealpha,
                        field="linecolor",
                    ),
                    linewidth=sealevel_data.linewidth,
                    label=sealevel_data.name,
                    zorder=sealevel_data.zorder,
                    markerfacecolor=markersettings["markerfacecolor"],
                    marker=markersettings["marker"],
                    markersize=markersettings["markersize"],
                    markeredgecolor=markersettings["markeredgecolor"],
                    markeredgewidth=markersettings["markeredgewidth"],
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

        self.update_ytick_labels(
            locations=[n / denominator for n in range(minimum_numerator, maximum_numerator + 1)],
            labels=[f"{n}/{denominator}" for n in range(minimum_numerator, maximum_numerator + 1)],
        )

    @property
    def _legend_handles(self) -> list[LegendHandle]:
        """Generated legend handles for sealevel sets."""
        handles: list[LegendHandle] = []
        handles.extend(self._get_sealevel_legend_handles())
        handles.extend(self._get_named_line_legend_handles())
        handles.extend(self._get_named_band_legend_handles())

        return handles
