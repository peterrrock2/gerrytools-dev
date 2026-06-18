from __future__ import annotations

import enum
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Final, Iterable, Mapping, Sequence, SupportsFloat, cast

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from numpy.typing import NDArray

from gerrytools.colors import resolve_color_and_alpha
from gerrytools.logging import get_logger
from gerrytools.plotting.data._categorical_distribution_base import CategoricalDistributionPlotBase
from gerrytools.plotting.data.options import BoxPlotOptions
from gerrytools.plotting.mpl.marker_options import PointMarkerOptions
from gerrytools.typing import Color, LegendHandle

logger = get_logger(__name__)


class _Unset(enum.Enum):
    """Sentinel distinguishing an omitted color kwarg from an explicit ``None``.

    Color kwargs default to this rather than ``None`` so a caller can pass
    ``facecolor=None`` (or ``edgecolor=None``) to mean "no fill/edge" (resolved
    to ``"none"``), while a genuinely omitted kwarg still falls back to the
    ``options`` default.
    """

    token = enum.auto()


_UNSET: Final = _Unset.token


# Field names a summary-stats mapping/DataFrame must provide for each category.
_REQUIRED_STAT_FIELDS: tuple[str, ...] = (
    "median",
    "lower_quartile",
    "upper_quartile",
    "lower_whisker",
    "upper_whisker",
)


def _as_float(value: object) -> float:
    """Coerce a scalar (Python or NumPy) value to a plain ``float``."""
    return float(cast("SupportsFloat | str", value))


def _coerce_fliers(value: object) -> tuple[float, ...]:
    """Coerce a fliers input into a tuple of floats.

    Accepts a sequence or NumPy array of values. ``None`` and lone scalars (for
    example a NaN produced by an absent DataFrame cell) are treated as "no
    fliers" so that summary-stats input without outliers parses cleanly.

    Args:
        value (object): Raw fliers value from a mapping or DataFrame cell.

    Returns:
        tuple[float, ...]: Outlier values, possibly empty.
    """
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        raise TypeError("fliers must be a sequence of numbers, not a string.")
    if isinstance(value, np.ndarray):
        return tuple(_as_float(flier) for flier in cast("Iterable[object]", value))
    if isinstance(value, Sequence):
        return tuple(_as_float(flier) for flier in value)
    # A lone scalar (e.g. NaN from a missing DataFrame cell) means "no fliers".
    return ()


@dataclass(frozen=True)
class BoxPlotSetData:
    """A set of boxplots to render for one model/series."""

    name: str
    scores_dict: dict[str, list[float]]
    facecolor: Color | None
    facealpha: float | None = None
    edgecolor: Color | None = "black"
    edgealpha: float | None = None
    edgewidth: float = 0.8
    percentiles: tuple[float, float] = (1, 99)
    showfliers: bool = False
    flier_options: PointMarkerOptions = field(default_factory=PointMarkerOptions)
    zorder: int = 1

    def __post_init__(self) -> None:
        lo, hi = self.percentiles
        lo = float(lo)
        hi = float(hi)
        if not (0.0 <= lo <= 100.0 and 0.0 <= hi <= 100.0):
            raise ValueError("percentiles must be within [0, 100].")
        if not (lo < hi):
            raise ValueError("percentiles must satisfy low < high.")

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
            owner=f"BoxPlotSetData {self.name}",
            logger=logger,
        )
        object.__setattr__(self, "facecolor", resolved_facecolor)
        object.__setattr__(self, "facealpha", resolved_alpha)

        resolved_edgecolor, resolved_edgealpha = resolve_color_and_alpha(
            self.edgecolor,
            alpha=self.edgealpha,
            allow_none=True,
            field="edgecolor",
            owner=f"BoxPlotSetData {self.name}",
            logger=logger,
        )
        object.__setattr__(self, "edgecolor", resolved_edgecolor)
        object.__setattr__(self, "edgealpha", resolved_edgealpha)

        if resolved_edgecolor.lower() == "none" and lw > 0:
            logger.log(
                level=logging.DEBUG,
                msg=(
                    f"For BoxPlotSetData {self.name}: edgecolor is 'none' but "
                    f"edgewidth is {lw}>0; setting edgewidth to 0."
                ),
            )
            lw = 0.0

        object.__setattr__(self, "edgewidth", lw)
        object.__setattr__(self, "zorder", int(self.zorder))


@dataclass(frozen=True)
class BoxPlotStats:
    """Precomputed summary statistics for a single box.

    Pass these to :meth:`BoxPlot.add_boxplot_stats_dataset` to draw a box from
    an already-computed five-number summary instead of from raw samples — useful
    when the underlying ensemble is too large to keep in memory or the statistics
    were computed elsewhere.

    Attributes:
        median (float): Median (Q2); the line drawn inside the box.
        lower_quartile (float): Lower quartile (Q1); the bottom of the box.
        upper_quartile (float): Upper quartile (Q3); the top of the box.
        lower_whisker (float): Lower whisker cap position.
        upper_whisker (float): Upper whisker cap position.
        mean (float | None): Optional mean. Rendered only when every box in the
            dataset provides one. Defaults to None.
        fliers (Sequence[float]): Optional outlier values, rendered when
            ``showfliers`` is enabled for the dataset. Defaults to empty.

    Raises:
        ValueError: If any required statistic is non-finite or the values do not
            satisfy ``lower_whisker <= lower_quartile <= median <= upper_quartile
            <= upper_whisker``.
    """

    median: float
    lower_quartile: float
    upper_quartile: float
    lower_whisker: float
    upper_whisker: float
    mean: float | None = None
    fliers: Sequence[float] = ()

    def __post_init__(self) -> None:
        median = _as_float(self.median)
        lower_quartile = _as_float(self.lower_quartile)
        upper_quartile = _as_float(self.upper_quartile)
        lower_whisker = _as_float(self.lower_whisker)
        upper_whisker = _as_float(self.upper_whisker)

        for field_name, field_value in (
            ("median", median),
            ("lower_quartile", lower_quartile),
            ("upper_quartile", upper_quartile),
            ("lower_whisker", lower_whisker),
            ("upper_whisker", upper_whisker),
        ):
            if not math.isfinite(field_value):
                raise ValueError(f"{field_name} must be a finite number; got {field_value!r}.")

        if not (lower_whisker <= lower_quartile <= median <= upper_quartile <= upper_whisker):
            raise ValueError(
                "Box plot statistics must satisfy lower_whisker <= lower_quartile <= median "
                "<= upper_quartile <= upper_whisker; got "
                f"lower_whisker={lower_whisker}, lower_quartile={lower_quartile}, "
                f"median={median}, upper_quartile={upper_quartile}, upper_whisker={upper_whisker}."
            )

        object.__setattr__(self, "median", median)
        object.__setattr__(self, "lower_quartile", lower_quartile)
        object.__setattr__(self, "upper_quartile", upper_quartile)
        object.__setattr__(self, "lower_whisker", lower_whisker)
        object.__setattr__(self, "upper_whisker", upper_whisker)

        if self.mean is not None:
            mean = _as_float(self.mean)
            if not math.isfinite(mean):
                raise ValueError(f"mean must be a finite number when provided; got {mean!r}.")
            object.__setattr__(self, "mean", mean)

        object.__setattr__(self, "fliers", tuple(_as_float(flier) for flier in self.fliers))

    def to_bxp_dict(self, label: str) -> dict[str, object]:
        """Convert to the per-box dict consumed by Matplotlib's ``Axes.bxp``.

        Args:
            label (str): Category label for the box.

        Returns:
            dict[str, object]: Matplotlib ``bxp`` statistics dict. The ``mean``
            key is included only when :attr:`mean` is set.
        """
        stats: dict[str, object] = {
            "label": label,
            "med": self.median,
            "q1": self.lower_quartile,
            "q3": self.upper_quartile,
            "whislo": self.lower_whisker,
            "whishi": self.upper_whisker,
            "fliers": list(self.fliers),
        }
        if self.mean is not None:
            stats["mean"] = self.mean
        return stats


@dataclass(frozen=True)
class BoxPlotStatsSetData:
    """A set of boxplots to render from precomputed summary statistics.

    Mirrors :class:`BoxPlotSetData` but carries a per-category
    :class:`BoxPlotStats` mapping instead of raw scores. There is no
    ``percentiles`` field because the whiskers are supplied directly.
    """

    name: str
    stats_dict: dict[str, BoxPlotStats]
    facecolor: Color | None
    facealpha: float | None = None
    edgecolor: Color | None = "black"
    edgealpha: float | None = None
    edgewidth: float = 0.8
    showfliers: bool = False
    flier_options: PointMarkerOptions = field(default_factory=PointMarkerOptions)
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
            owner=f"BoxPlotStatsSetData {self.name}",
            logger=logger,
        )
        object.__setattr__(self, "facecolor", resolved_facecolor)
        object.__setattr__(self, "facealpha", resolved_alpha)

        resolved_edgecolor, resolved_edgealpha = resolve_color_and_alpha(
            self.edgecolor,
            alpha=self.edgealpha,
            allow_none=True,
            field="edgecolor",
            owner=f"BoxPlotStatsSetData {self.name}",
            logger=logger,
        )
        object.__setattr__(self, "edgecolor", resolved_edgecolor)
        object.__setattr__(self, "edgealpha", resolved_edgealpha)

        if resolved_edgecolor.lower() == "none" and lw > 0:
            logger.log(
                level=logging.DEBUG,
                msg=(
                    f"For BoxPlotStatsSetData {self.name}: edgecolor is 'none' but "
                    f"edgewidth is {lw}>0; setting edgewidth to 0."
                ),
            )
            lw = 0.0

        object.__setattr__(self, "edgewidth", lw)
        object.__setattr__(self, "zorder", int(self.zorder))


class BoxPlot(CategoricalDistributionPlotBase):
    """Create grouped boxplot comparison figures across categories.

    Add data either from raw samples via :meth:`add_boxplot_dataset` or from a
    precomputed five-number summary via :meth:`add_boxplot_stats_dataset`. Both
    kinds of sets can be combined on a single figure and share the grouped layout.
    """

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
        boxplot_width_scale: float = 0.8,
        boxplot_group_width: float = 0.7,
    ) -> None:
        """Initialize a BoxPlot.

        Toggle the per-group vertical guide lines via
        :meth:`enable_boxplot_group_vlines` / :meth:`disable_boxplot_group_vlines`
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
            group_width=boxplot_group_width,
            width_scale=boxplot_width_scale,
            include_group_vlines=False,
        )
        self._boxplot_data_list: list[BoxPlotSetData | BoxPlotStatsSetData] = []

    def enable_boxplot_group_vlines(self) -> None:
        """Show vertical guide lines at the center of each category group."""
        self._include_group_vlines = True

    def disable_boxplot_group_vlines(self) -> None:
        """Hide the per-category vertical guide lines (the default)."""
        self._include_group_vlines = False

    @property
    def boxplot_group_width(self) -> float:
        """Width allocated to each category group."""
        return self.group_width

    @boxplot_group_width.setter
    def boxplot_group_width(self, value: float) -> None:
        """Set width allocated to each category group.

        Args:
            value (float): Group width in x-axis data units.

        Returns:
            None
        """
        self.group_width = float(value)

    @property
    def boxplot_width_scale(self) -> float:
        """Scale factor for each per-set box width inside a group."""
        return self.width_scale

    @boxplot_width_scale.setter
    def boxplot_width_scale(self, value: float) -> None:
        """Set per-set width scaling within each category group.

        Args:
            value (float): Width scale multiplier.

        Returns:
            None
        """
        self.width_scale = float(value)

    @staticmethod
    def _convert_boxplot_data_to_dictionary(
        scores: (
            Mapping[str, Sequence[int | float]]
            | Sequence[int | float]
            | Sequence[Sequence[int | float]]
            | pd.DataFrame
            | NDArray
        ),
        scores_labels: list[str] | None = None,
    ) -> dict[str, list[float]]:
        """Convert boxplot input to a dictionary mapping labels to score lists.

        Args:
            scores (dict[str, list[float]] | list[float] | list[list[float]] | pd.DataFrame):
                Boxplot distribution input.
            scores_labels (list[str] | None, optional): Labels for list-based input.
                Defaults to None.

        Returns:
            dict[str, list[float]]: Category label to score-list mapping.
        """
        return CategoricalDistributionPlotBase._convert_distribution_data_to_dictionary(
            scores,
            scores_labels,
        )

    def add_boxplot_dataset(
        self,
        scores: (
            Mapping[str, Sequence[float]]
            | Sequence[float]
            | Sequence[Sequence[float]]
            | pd.DataFrame
            | NDArray
        ),
        name: str | None = None,
        *,
        scores_labels: list[str] | None = None,
        options: BoxPlotOptions | None = None,
        facecolor: Color | None | _Unset = _UNSET,
        facealpha: float | None = None,
        edgecolor: Color | None | _Unset = _UNSET,
        edgealpha: float | None = None,
        edgewidth: float | None = None,
        percentiles: tuple[float, float] | None = None,
        showfliers: bool | None = None,
        flier_options: PointMarkerOptions | None = None,
        add_extra_labels: bool = False,
        zorder: int | None = None,
    ) -> None:
        """Add one boxplot dataset (one box per category) to the figure.

        Args:
            scores (dict[str, list[float]] | list[float] | list[list[float]] | pd.DataFrame):
                Distribution input by category.
            scores_labels (list[str] | None, optional): Labels for list-based input.
                Defaults to None.
            name (str | None, optional): Legend label for the dataset. Defaults to None.
            options (BoxPlotOptions | None, optional): Base styling whose values are used
                for any styling argument left as None. Defaults to None.
            facecolor (Color | None, optional): Box fill color. Pass ``None`` for an
                unfilled (transparent) box. Omit to use the ``options`` default ``"denim"``.
            facealpha (float | None, optional): Box fill alpha override. Defaults to None.
            edgecolor (Color | None, optional): Box edge color. Pass ``None`` for no edge.
                Omit to use the ``options`` default ``"black"``.
            edgealpha (float | None, optional): Box edge alpha override. Defaults to None.
            edgewidth (float, optional): Box edge width. Defaults to ``0.8``.
            percentiles (tuple[float, float], optional): Lower/upper whisker percentiles.
                Defaults to ``(1, 99)``.
            showfliers (bool, optional): Whether to show outlier markers. Defaults to False.
            flier_options (PointMarkerOptions | None, optional): Outlier marker options.
                Defaults to None.
            add_extra_labels (bool, optional): Whether to merge unseen incoming labels into
                existing category labels. Defaults to False.
            zorder (int, optional): Draw order for the dataset. Defaults to ``1``.

        Returns:
            None
        """
        base = options if options is not None else BoxPlotOptions()
        # facecolor/edgecolor use the _UNSET sentinel so an explicit None reaches
        # resolve_color_and_alpha (which maps it to "none"); only a truly omitted
        # kwarg falls back to the options default.
        resolved_facecolor = base.facecolor if facecolor is _UNSET else facecolor
        resolved_facealpha = facealpha if facealpha is not None else base.facealpha
        resolved_edgecolor = base.edgecolor if edgecolor is _UNSET else edgecolor
        resolved_edgealpha = edgealpha if edgealpha is not None else base.edgealpha
        resolved_edgewidth = edgewidth if edgewidth is not None else base.edgewidth
        resolved_percentiles = percentiles if percentiles is not None else base.percentiles
        resolved_showfliers = showfliers if showfliers is not None else base.showfliers
        resolved_flier_options = flier_options if flier_options is not None else base.flier_options
        resolved_zorder = zorder if zorder is not None else base.zorder

        scores_dict = self._convert_boxplot_data_to_dictionary(scores, scores_labels)
        self._sync_labels(
            list(scores_dict.keys()),
            add_extra_labels=add_extra_labels,
            item_name="boxplot set",
        )

        set_name = name or f"Set {len(self._boxplot_data_list) + 1}"
        self._boxplot_data_list.append(
            BoxPlotSetData(
                scores_dict=scores_dict,
                name=set_name,
                facecolor=resolved_facecolor,
                facealpha=resolved_facealpha,
                edgecolor=resolved_edgecolor,
                edgealpha=resolved_edgealpha,
                edgewidth=resolved_edgewidth,
                percentiles=resolved_percentiles,
                showfliers=resolved_showfliers,
                flier_options=resolved_flier_options,
                zorder=resolved_zorder,
            )
        )
        self._claim_legend_if_named(name)

    @staticmethod
    def _box_plot_stats_from_mapping(
        values: Mapping[str, object],
        *,
        category: str,
    ) -> BoxPlotStats:
        """Build a :class:`BoxPlotStats` from a mapping of field names to values.

        Args:
            values (Mapping[str, object]): Field-name to value mapping (e.g. a dict
                or a DataFrame row converted via ``Series.to_dict``).
            category (str): Category label, used in error messages.

        Returns:
            BoxPlotStats: The parsed statistics.

        Raises:
            ValueError: If any required field is missing.
        """
        missing = [name for name in _REQUIRED_STAT_FIELDS if name not in values]
        if missing:
            raise ValueError(
                f"Box plot stats for category {category!r} is missing required "
                f"field(s): {', '.join(missing)}."
            )

        raw_mean = values.get("mean")
        if raw_mean is None:
            mean: float | None = None
        else:
            mean_value = _as_float(raw_mean)
            # A NaN mean (e.g. an absent DataFrame cell) means "no mean".
            mean = None if math.isnan(mean_value) else mean_value

        return BoxPlotStats(
            median=_as_float(values["median"]),
            lower_quartile=_as_float(values["lower_quartile"]),
            upper_quartile=_as_float(values["upper_quartile"]),
            lower_whisker=_as_float(values["lower_whisker"]),
            upper_whisker=_as_float(values["upper_whisker"]),
            mean=mean,
            fliers=_coerce_fliers(values.get("fliers", ())),
        )

    @staticmethod
    def _convert_boxplot_stats_to_dictionary(
        stats: Mapping[str, BoxPlotStats | Mapping[str, object]] | pd.DataFrame,
    ) -> dict[str, BoxPlotStats]:
        """Normalize summary-stats input into a ``{category: BoxPlotStats}`` dict.

        Args:
            stats (Mapping[str, BoxPlotStats | Mapping[str, object]] | pd.DataFrame):
                Per-category statistics. A DataFrame uses its index as categories
                and columns as stat field names.

        Returns:
            dict[str, BoxPlotStats]: Category label to statistics mapping.

        Raises:
            TypeError: If ``stats`` (or one of its values) is an unsupported type.
        """
        if isinstance(stats, pd.DataFrame):
            return {
                str(category): BoxPlot._box_plot_stats_from_mapping(
                    cast("Mapping[str, object]", row.to_dict()),
                    category=str(category),
                )
                for category, row in stats.iterrows()
            }

        if isinstance(stats, Mapping):
            converted: dict[str, BoxPlotStats] = {}
            for category, value in stats.items():
                category_label = str(category)
                if isinstance(value, BoxPlotStats):
                    converted[category_label] = value
                elif isinstance(value, Mapping):
                    converted[category_label] = BoxPlot._box_plot_stats_from_mapping(
                        cast("Mapping[str, object]", value),
                        category=category_label,
                    )
                else:
                    raise TypeError(
                        f"Box plot stats for category {category_label!r} must be a BoxPlotStats "
                        f"or a mapping of field names to values; got {type(value).__name__}."
                    )
            return converted

        raise TypeError(
            "stats must be a Mapping[str, BoxPlotStats | Mapping[str, float]] or a pandas "
            "DataFrame."
        )

    def add_boxplot_stats_dataset(
        self,
        stats: Mapping[str, BoxPlotStats | Mapping[str, object]] | pd.DataFrame,
        name: str | None = None,
        *,
        options: BoxPlotOptions | None = None,
        facecolor: Color | None | _Unset = _UNSET,
        facealpha: float | None = None,
        edgecolor: Color | None | _Unset = _UNSET,
        edgealpha: float | None = None,
        edgewidth: float | None = None,
        showfliers: bool | None = None,
        flier_options: PointMarkerOptions | None = None,
        add_extra_labels: bool = False,
        zorder: int | None = None,
    ) -> None:
        """Add one boxplot dataset from precomputed summary statistics.

        Use this instead of :meth:`add_boxplot_dataset` when you already have a
        five-number summary per category (median, quartiles, whiskers) rather than
        the raw samples. Boxes are drawn via Matplotlib's ``Axes.bxp``. Stats and
        raw datasets can be mixed on the same figure; all sets share the grouped
        layout and are positioned together.

        Args:
            stats (Mapping[str, BoxPlotStats | Mapping[str, object]] | pd.DataFrame):
                Per-category statistics. Each value may be a :class:`BoxPlotStats`,
                a mapping of field names (``median``, ``lower_quartile``,
                ``upper_quartile``, ``lower_whisker``, ``upper_whisker``, and the
                optional ``mean`` / ``fliers``) to values, or — for a DataFrame —
                a row whose columns are those field names (index = categories).
            name (str | None, optional): Legend label for the dataset. Defaults to None.
            options (BoxPlotOptions | None, optional): Base styling whose values are
                used for any styling argument left as None. The ``percentiles`` field
                is ignored because whiskers are supplied directly. Defaults to None.
            facecolor (Color | None, optional): Box fill color. Pass ``None`` for an
                unfilled (transparent) box. Omit to use the ``options`` default ``"denim"``.
            facealpha (float | None, optional): Box fill alpha override. Defaults to None.
            edgecolor (Color | None, optional): Box edge color. Pass ``None`` for no edge.
                Omit to use the ``options`` default ``"black"``.
            edgealpha (float | None, optional): Box edge alpha override. Defaults to None.
            edgewidth (float | None, optional): Box edge width. Defaults to ``0.8``.
            showfliers (bool | None, optional): Whether to render outlier markers from
                each box's ``fliers``. Defaults to False.
            flier_options (PointMarkerOptions | None, optional): Outlier marker options.
                Defaults to None.
            add_extra_labels (bool, optional): Whether to merge unseen incoming labels
                into existing category labels. Defaults to False.
            zorder (int | None, optional): Draw order for the dataset. Defaults to ``1``.

        Returns:
            None

        Raises:
            ValueError: If ``stats`` is empty.
        """
        base = options if options is not None else BoxPlotOptions()
        # facecolor/edgecolor use the _UNSET sentinel so an explicit None reaches
        # resolve_color_and_alpha (which maps it to "none"); only a truly omitted
        # kwarg falls back to the options default.
        resolved_facecolor = base.facecolor if facecolor is _UNSET else facecolor
        resolved_facealpha = facealpha if facealpha is not None else base.facealpha
        resolved_edgecolor = base.edgecolor if edgecolor is _UNSET else edgecolor
        resolved_edgealpha = edgealpha if edgealpha is not None else base.edgealpha
        resolved_edgewidth = edgewidth if edgewidth is not None else base.edgewidth
        resolved_showfliers = showfliers if showfliers is not None else base.showfliers
        resolved_flier_options = flier_options if flier_options is not None else base.flier_options
        resolved_zorder = zorder if zorder is not None else base.zorder

        stats_dict = self._convert_boxplot_stats_to_dictionary(stats)
        if len(stats_dict) == 0:
            raise ValueError("stats is empty; provide statistics for at least one category.")

        self._sync_labels(
            list(stats_dict.keys()),
            add_extra_labels=add_extra_labels,
            item_name="boxplot stats set",
        )

        set_name = name or f"Set {len(self._boxplot_data_list) + 1}"
        self._boxplot_data_list.append(
            BoxPlotStatsSetData(
                name=set_name,
                stats_dict=stats_dict,
                facecolor=resolved_facecolor,
                facealpha=resolved_facealpha,
                edgecolor=resolved_edgecolor,
                edgealpha=resolved_edgealpha,
                edgewidth=resolved_edgewidth,
                showfliers=resolved_showfliers,
                flier_options=resolved_flier_options,
                zorder=resolved_zorder,
            )
        )
        self._claim_legend_if_named(name)

    @property
    def _boxplot_centers(self) -> np.ndarray:
        """Calculate x-axis centers for each boxplot category."""
        return self._category_centers

    def _draw_boxplot_group_vlines(self) -> None:
        """Draw vertical lines at the center of boxplot groups."""
        self._draw_group_vlines()

    def _draw_boxplot(self) -> None:
        """Draw all boxplot sets (raw-data and stats-based) on the plot."""
        n_boxplot_sets = len(self._boxplot_data_list)
        if n_boxplot_sets == 0:  # pragma: no cover - _build_plot raises first
            return  # pragma: no cover

        if (
            self._labels is None
        ):  # pragma: no cover - _build_plot raises first if labels are missing
            return  # pragma: no cover

        centers = self._boxplot_centers
        box_width = self.boxplot_group_width / n_boxplot_sets
        offsets = (np.arange(n_boxplot_sets) - (n_boxplot_sets - 1) / 2.0) * box_width
        widths = box_width * self.boxplot_width_scale

        for k, set_data in enumerate(self._boxplot_data_list):
            positions_all = centers + offsets[k]
            if isinstance(set_data, BoxPlotSetData):
                drawn = self._draw_raw_box_set(set_data, positions_all, widths)
            else:
                drawn = self._draw_stats_box_set(set_data, positions_all, widths)

            if drawn is not None:
                self._style_box_set(drawn, set_data)

    def _draw_raw_box_set(
        self,
        set_data: BoxPlotSetData,
        positions_all: np.ndarray,
        widths: float,
    ) -> dict[str, Any] | None:
        """Draw one raw-data boxplot set via ``Axes.boxplot``.

        Args:
            set_data (BoxPlotSetData): The raw-data set to draw.
            positions_all (np.ndarray): X positions for every category center.
            widths (float): Box width.

        Returns:
            dict[str, Any] | None: The Matplotlib artist dict, or None when the set
            has no non-empty categories to draw.
        """
        assert self._labels is not None
        data_k: list[list[float]] = []
        pos_k: list[float] = []
        for lab, x in zip(self._labels, positions_all, strict=True):
            vals = set_data.scores_dict.get(lab, [])
            if vals is None or len(vals) == 0:
                continue
            data_k.append(list(vals))
            pos_k.append(float(x))

        if len(data_k) == 0:
            return None

        return cast(
            "dict[str, Any]",
            self._ax.boxplot(
                data_k,
                positions=pos_k,
                widths=widths,
                patch_artist=True,
                showfliers=set_data.showfliers,
                whis=set_data.percentiles,
                # Cast to satisfy the type-checker: Matplotlib stubs expect
                # a plain ``dict[str, object]`` for ``flierprops``.
                flierprops=cast(
                    dict[str, object],
                    set_data.flier_options.to_mpl_settings_dict(),
                ),
            ),
        )

    def _draw_stats_box_set(
        self,
        set_data: BoxPlotStatsSetData,
        positions_all: np.ndarray,
        widths: float,
    ) -> dict[str, Any] | None:
        """Draw one summary-stats boxplot set via ``Axes.bxp``.

        Means are rendered only when every drawn box in the set provides one.

        Args:
            set_data (BoxPlotStatsSetData): The stats-based set to draw.
            positions_all (np.ndarray): X positions for every category center.
            widths (float): Box width.

        Returns:
            dict[str, Any] | None: The Matplotlib artist dict, or None when the set
            has no categories to draw.
        """
        assert self._labels is not None
        bxp_stats: list[dict[str, object]] = []
        pos_k: list[float] = []
        means_present: list[bool] = []
        for lab, x in zip(self._labels, positions_all, strict=True):
            stat = set_data.stats_dict.get(lab)
            if stat is None:
                continue
            bxp_stats.append(stat.to_bxp_dict(lab))
            pos_k.append(float(x))
            means_present.append(stat.mean is not None)

        if len(bxp_stats) == 0:
            return None

        return cast(
            "dict[str, Any]",
            self._ax.bxp(
                bxp_stats,
                positions=pos_k,
                widths=widths,
                patch_artist=True,
                showfliers=set_data.showfliers,
                showmeans=all(means_present),
                # Cast to satisfy the type-checker: Matplotlib stubs expect
                # a plain ``dict[str, object]`` for ``flierprops``.
                flierprops=cast(
                    dict[str, object],
                    set_data.flier_options.to_mpl_settings_dict(),
                ),
            ),
        )

    def _style_box_set(
        self,
        drawn: dict[str, Any],
        set_data: BoxPlotSetData | BoxPlotStatsSetData,
    ) -> None:
        """Apply gerrytools styling to the artists of one drawn box set.

        ``Axes.boxplot`` and ``Axes.bxp`` return the same dict of artist lists
        keyed by component name (boxes, whiskers, caps, medians, means, fliers),
        so both raw-data and stats-based sets are styled identically here.

        Args:
            drawn (dict[str, Any]): Matplotlib artist dict from boxplot/bxp.
            set_data (BoxPlotSetData | BoxPlotStatsSetData): The set's styling.

        Returns:
            None
        """
        for key in ("boxes", "whiskers", "caps", "medians", "means", "fliers"):
            for artist in drawn.get(key, []):
                self._artists.track(artist)

        facecolor = self._resolved_rgba(set_data.facecolor, set_data.facealpha, field="facecolor")
        edgecolor = self._resolved_rgba(set_data.edgecolor, set_data.edgealpha, field="edgecolor")

        for patch in drawn["boxes"]:
            patch.set_facecolor(facecolor)
            patch.set_linewidth(set_data.edgewidth)
            patch.set_edgecolor(edgecolor)
            patch.set_zorder(set_data.zorder)

        for key in ("whiskers", "caps", "medians", "means"):
            for artist in drawn.get(key, []):
                artist.set_color(edgecolor)
                artist.set_linewidth(set_data.edgewidth)
                artist.set_zorder(set_data.zorder)

        for artist in drawn.get("fliers", []):
            artist.set_zorder(set_data.zorder)

    def _build_plot(self) -> None:
        """Build the boxplot figure."""
        if self._labels is None or len(self._labels) == 0:
            raise ValueError("No labels defined yet.")

        if len(self._boxplot_data_list) == 0:
            raise ValueError("No boxplot sets added yet.")

        self._draw_boxplot()
        self._draw_pointset(self._boxplot_centers)

        if self._include_group_vlines:
            self._draw_boxplot_group_vlines()

    def _get_boxplot_legend_handles(self) -> list[LegendHandle]:
        """Generate legend handles for boxplot sets."""
        handles: list[LegendHandle] = []

        for boxplot_data in self._boxplot_data_list:
            handles.append(
                Patch(
                    facecolor=self._resolved_rgba(
                        boxplot_data.facecolor,
                        boxplot_data.facealpha,
                        field="facecolor",
                    ),
                    edgecolor=self._resolved_rgba(
                        boxplot_data.edgecolor,
                        boxplot_data.edgealpha,
                        field="edgecolor",
                    ),
                    label=boxplot_data.name,
                )
            )

        return handles

    @property
    def _legend_handles(self) -> list[LegendHandle]:
        """Generated legend handles for boxplot and point sets."""
        handles: list[LegendHandle] = []

        handles.extend(self._get_boxplot_legend_handles())
        handles.extend(self._get_pointset_legend_handles())
        handles.extend(self._get_named_line_legend_handles())
        handles.extend(self._get_named_band_legend_handles())

        return handles
