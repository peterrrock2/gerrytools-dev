from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch

from gerrytools.colors import resolve_color_and_alpha
from gerrytools.logging import get_logger
from gerrytools.plotting.data._categorical_distribution_base import CategoricalDistributionPlotBase
from gerrytools.plotting.data.options import ViolinPlotOptions
from gerrytools.typing import Color, LegendHandle

logger = get_logger(__name__)


@dataclass(frozen=True)
class ViolinPlotSetData:
    """A set of violinplots to render for one model/series."""

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


class ViolinPlot(CategoricalDistributionPlotBase):
    """Create grouped violinplot comparison figures across categories."""

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
        violinplot_width_scale: float = 0.8,
        violinplot_group_width: float = 0.7,
    ) -> None:
        """Initialize a ViolinPlot.

        Toggle the per-group vertical guide lines via
        :meth:`enable_violinplot_group_vlines` / :meth:`disable_violinplot_group_vlines`
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
            group_width=violinplot_group_width,
            width_scale=violinplot_width_scale,
            include_group_vlines=False,
        )
        self._violinplot_data_list: list[ViolinPlotSetData] = []

    def enable_violinplot_group_vlines(self) -> None:
        """Show vertical guide lines at the center of each category group."""
        self._include_group_vlines = True

    def disable_violinplot_group_vlines(self) -> None:
        """Hide the per-category vertical guide lines (the default)."""
        self._include_group_vlines = False

    @property
    def violinplot_group_width(self) -> float:
        """Width allocated to each category group."""
        return self.group_width

    @violinplot_group_width.setter
    def violinplot_group_width(self, value: float) -> None:
        """Set width allocated to each category group.

        Args:
            value (float): Group width in x-axis data units.

        Returns:
            None
        """
        self.group_width = float(value)

    @property
    def violinplot_width_scale(self) -> float:
        """Scale factor for each per-set violin width inside a group."""
        return self.width_scale

    @violinplot_width_scale.setter
    def violinplot_width_scale(self, value: float) -> None:
        """Set per-set violin width scaling within each category group.

        Args:
            value (float): Width scale multiplier.

        Returns:
            None
        """
        self.width_scale = float(value)

    @staticmethod
    def _convert_violinplot_data_to_dictionary(
        scores: dict[str, list[float]] | list[float] | list[list[float]] | pd.DataFrame,
        scores_labels: list[str] | None = None,
    ) -> dict[str, list[float]]:
        """Convert violinplot input to a dictionary mapping labels to score lists.

        Args:
            scores (dict[str, list[float]] | list[float] | list[list[float]] | pd.DataFrame):
                Violinplot distribution input.
            scores_labels (list[str] | None, optional): Labels for list-based input.
                Defaults to None.

        Returns:
            dict[str, list[float]]: Category label to score-list mapping.
        """
        return CategoricalDistributionPlotBase._convert_distribution_data_to_dictionary(
            scores,
            scores_labels,
        )

    def add_violinplot_datasets(
        self,
        scores: dict[str, list[float]] | list[float] | list[list[float]] | pd.DataFrame,
        name: str | None = None,
        *,
        scores_labels: list[str] | None = None,
        options: ViolinPlotOptions | None = None,
        facecolor: Color | None = None,
        facealpha: float | None = None,
        edgecolor: Color | None = None,
        edgealpha: float | None = None,
        edgewidth: float | None = None,
        add_extra_labels: bool = False,
        zorder: int | None = None,
    ) -> None:
        """Add one violinplot dataset (one violin per category) to the figure.

        Args:
            scores (dict[str, list[float]] | list[float] | list[list[float]] | pd.DataFrame):
                Distribution input by category.
            scores_labels (list[str] | None, optional): Labels for list-based input.
                Defaults to None.
            name (str | None, optional): Legend label for the dataset. Defaults to None.
            facecolor (Color, optional): Violin fill color. Defaults to ``"denim"``.
            facealpha (float | None, optional): Violin fill alpha override. Defaults to None.
            edgecolor (Color, optional): Violin edge color. Defaults to ``"black"``.
            edgealpha (float | None, optional): Violin edge alpha override. Defaults to None.
            edgewidth (float, optional): Violin edge width. Defaults to ``0.8``.
            add_extra_labels (bool, optional): Whether to merge unseen incoming labels into
                existing category labels. Defaults to False.
            zorder (int, optional): Draw order for the dataset. Defaults to ``1``.

        Returns:
            None
        """
        base = options if options is not None else ViolinPlotOptions()
        resolved_facecolor = facecolor if facecolor is not None else base.facecolor
        resolved_facealpha = facealpha if facealpha is not None else base.facealpha
        resolved_edgecolor = edgecolor if edgecolor is not None else base.edgecolor
        resolved_edgealpha = edgealpha if edgealpha is not None else base.edgealpha
        resolved_edgewidth = edgewidth if edgewidth is not None else base.edgewidth
        resolved_zorder = zorder if zorder is not None else base.zorder

        scores_dict = self._convert_violinplot_data_to_dictionary(scores, scores_labels)
        self._sync_labels(
            list(scores_dict.keys()),
            add_extra_labels=add_extra_labels,
            item_name="violinplot set",
        )

        set_name = name or f"Set {len(self._violinplot_data_list) + 1}"
        self._violinplot_data_list.append(
            ViolinPlotSetData(
                scores_dict=scores_dict,
                name=set_name,
                facecolor=resolved_facecolor,
                facealpha=resolved_facealpha,
                edgecolor=resolved_edgecolor,
                edgealpha=resolved_edgealpha,
                edgewidth=resolved_edgewidth,
                zorder=resolved_zorder,
            )
        )

    @property
    def _violinplot_centers(self) -> np.ndarray:
        """Calculate x-axis centers for each violinplot category."""
        return self._category_centers

    def _draw_violinplot_group_vlines(self) -> None:
        """Draw vertical lines at the center of violinplot groups."""
        self._draw_group_vlines()

    def _draw_violinplot(self) -> None:
        """Draw the violinplots on the plot."""
        n_violinplot_sets = len(self._violinplot_data_list)

        centers = self._violinplot_centers
        violin_width = self.violinplot_group_width / n_violinplot_sets
        offsets = (np.arange(n_violinplot_sets) - (n_violinplot_sets - 1) / 2.0) * violin_width
        widths = violin_width * self.violinplot_width_scale

        for k, violinplot_data in enumerate(self._violinplot_data_list):
            pos_k_all = centers + offsets[k]
            data_k: list[list[float]] = []
            pos_k: list[float] = []

            assert self._labels is not None, (
                "Internal error: _labels should have been set by _sync_labels when adding "
                "violinplot data."
            )
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

            bodies = cast(list[PolyCollection], vp.get("bodies", []))
            for patch in bodies:
                patch.set_alpha(None)
                patch.set_facecolor(
                    self._resolved_rgba(
                        violinplot_data.facecolor,
                        violinplot_data.facealpha,
                        field="facecolor",
                    )
                )
                patch.set_linewidth(violinplot_data.edgewidth)
                patch.set_edgecolor(
                    self._resolved_rgba(
                        violinplot_data.edgecolor,
                        violinplot_data.edgealpha,
                        field="edgecolor",
                    )
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
        if self._include_group_vlines:
            self._draw_violinplot_group_vlines()

    def _get_violinplot_legend_handles(self) -> list[LegendHandle]:
        """Generate legend handles for violinplot sets."""
        handles: list[LegendHandle] = []

        for violinplot_data in self._violinplot_data_list:
            handles.append(
                Patch(
                    facecolor=self._resolved_rgba(
                        violinplot_data.facecolor,
                        violinplot_data.facealpha,
                        field="facecolor",
                    ),
                    edgecolor=self._resolved_rgba(
                        violinplot_data.edgecolor,
                        violinplot_data.edgealpha,
                        field="edgecolor",
                    ),
                    label=violinplot_data.name,
                )
            )

        return handles

    @property
    def _legend_handles(self) -> list[LegendHandle]:
        """Generated legend handles for violinplot and point sets."""
        handles: list[LegendHandle] = []

        handles.extend(self._get_violinplot_legend_handles())
        handles.extend(self._get_pointset_legend_handles())
        handles.extend(self._get_named_line_legend_handles())
        handles.extend(self._get_named_band_legend_handles())

        return handles
