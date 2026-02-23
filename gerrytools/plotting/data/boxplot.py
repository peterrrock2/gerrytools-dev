from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from gerrytools.colors import resolve_color_and_alpha
from gerrytools.logging import get_logger
from gerrytools.plotting.data._categorical_distribution_base import CategoricalDistributionPlotBase
from gerrytools.plotting.mpl.marker_options import PointMarkerOptions
from gerrytools.typing import Color

logger = get_logger(__name__)


@dataclass(frozen=True)
class BoxPlotSetData:
    """A set of boxplots to render for one model/series."""

    name: str
    scores_dict: dict[str, list[float]]
    facecolor: Color
    facealpha: float | None = None
    edgecolor: Color = "black"
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


class BoxPlot(CategoricalDistributionPlotBase):
    """Create grouped boxplot comparison figures across categories."""

    def __init__(
        self,
        figure_size: tuple[float, float] = (10, 6),
        dpi: int = 300,
        *,
        include_legend: bool = True,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        boxplot_width_scale: float = 0.8,
        boxplot_group_width: float = 0.7,
        include_boxplot_group_vlines: bool = False,
    ) -> None:
        super().__init__(
            figure_size=figure_size,
            dpi=dpi,
            include_legend=include_legend,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            group_width=boxplot_group_width,
            width_scale=boxplot_width_scale,
            include_group_vlines=include_boxplot_group_vlines,
        )
        self._boxplot_data_list: list[BoxPlotSetData] = []

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
        scores: dict[str, list[float]] | list[float] | list[list[float]] | pd.DataFrame,
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

    def add_boxplot_datasets(
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
        percentiles: tuple[float, float] = (1, 99),
        showfliers: bool = False,
        flier_options: PointMarkerOptions | None = None,
        add_extra_labels: bool = False,
        zorder: int = 1,
    ) -> None:
        """Add one boxplot dataset (one box per category) to the figure.

        Args:
            scores (dict[str, list[float]] | list[float] | list[list[float]] | pd.DataFrame):
                Distribution input by category.
            scores_labels (list[str] | None, optional): Labels for list-based input.
                Defaults to None.
            name (str | None, optional): Legend label for the dataset. Defaults to None.
            facecolor (Color, optional): Box fill color. Defaults to ``"denim"``.
            facealpha (float | None, optional): Box fill alpha override. Defaults to None.
            edgecolor (Color, optional): Box edge color. Defaults to ``"black"``.
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
        if flier_options is None:
            flier_options = PointMarkerOptions()

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
                facecolor=facecolor,
                facealpha=facealpha,
                edgecolor=edgecolor,
                edgealpha=edgealpha,
                edgewidth=edgewidth,
                percentiles=percentiles,
                showfliers=showfliers,
                flier_options=flier_options,
                zorder=zorder,
            )
        )

    @property
    def _boxplot_centers(self) -> np.ndarray:
        """Calculate x-axis centers for each boxplot category."""
        return self._category_centers

    def _draw_boxplot_group_vlines(self) -> None:
        """Draw vertical lines at the center of boxplot groups."""
        self._draw_group_vlines()

    def _draw_boxplot(self) -> None:
        """Draw the boxplots on the plot."""
        n_boxplot_sets = len(self._boxplot_data_list)

        centers = self._boxplot_centers
        box_width = self.boxplot_group_width / n_boxplot_sets
        offsets = (np.arange(n_boxplot_sets) - (n_boxplot_sets - 1) / 2.0) * box_width
        widths = box_width * self.boxplot_width_scale

        for k, boxplot_data in enumerate(self._boxplot_data_list):
            pos_k_all = centers + offsets[k]
            data_k: list[list[float]] = []
            pos_k: list[float] = []

            if self._labels is None:
                continue

            for lab, x in zip(self._labels, pos_k_all, strict=True):
                vals = boxplot_data.scores_dict.get(lab, [])
                if vals is None or len(vals) == 0:
                    continue
                data_k.append(list(vals))
                pos_k.append(float(x))

            if len(data_k) == 0:
                continue

            bp = self._ax.boxplot(
                data_k,
                positions=pos_k,
                widths=widths,
                patch_artist=True,
                showfliers=boxplot_data.showfliers,
                whis=boxplot_data.percentiles,
                flierprops=boxplot_data.flier_options.to_mpl_settings_dict(),
            )

            edgecolor = self._resolved_rgba(
                boxplot_data.edgecolor,
                boxplot_data.edgealpha,
                field="edgecolor",
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(
                    self._resolved_rgba(
                        boxplot_data.facecolor,
                        boxplot_data.facealpha,
                        field="facecolor",
                    )
                )
                patch.set_linewidth(boxplot_data.edgewidth)
                patch.set_edgecolor(edgecolor)
                patch.set_zorder(boxplot_data.zorder)

            for key in ("whiskers", "caps", "medians", "means"):
                for artist in bp.get(key, []):
                    artist.set_color(edgecolor)
                    artist.set_linewidth(boxplot_data.edgewidth)
                    artist.set_zorder(boxplot_data.zorder)

            for artist in bp.get("fliers", []):
                artist.set_zorder(boxplot_data.zorder)

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

    def _get_boxplot_legend_handles(self) -> list[Any]:
        """Generate legend handles for boxplot sets."""
        handles: list[Any] = []

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
    def _legend_handles(self) -> list[Any]:
        """Generated legend handles for boxplot and point sets."""
        handles: list[Any] = []

        handles.extend(self._get_boxplot_legend_handles())
        handles.extend(self._get_pointset_legend_handles())
        handles.extend(self._get_named_line_legend_handles())
        handles.extend(self._get_named_band_legend_handles())

        return handles
