"""`_Annotations` — collected vlines/hlines/bands/arrows owned by `GerryPlotBase`.

Internal module. End users still call `plot.add_vertical_lines(...)` etc.;
those methods on `GerryPlotBase` delegate to an `_Annotations` instance held
on the plot. The point of this module is to localize the state-management +
rendering loop for "decorations on top of a plot" so the base class no longer
mixes plot orchestration with annotation bookkeeping.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Callable

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from gerrytools.plotting.data._additional_renderers import _AnnotationArrowRenderer
from gerrytools.plotting.data._gerryplot_dataclasses import (
    ArrowData,
    BandData,
    LineData,
)

# A color resolver: ``(color, alpha=None, *, field="color") -> RGBA``. The
# annotations module is decoupled from `GerryPlotBase` and is handed one of
# these so it can resolve colors without holding a back-reference.
ColorResolver = Callable[..., tuple[float, float, float, float]]


@dataclass
class _Annotations:
    """Mutable holder for all "decoration" annotations on a plot.

    Five fields, one per decoration kind. Each ``add_*`` / ``clear_*`` method
    on `GerryPlotBase` is a thin facade that mutates one of these lists, and
    `apply()` walks them at render time.
    """

    vertical_lines: list[LineData] = field(default_factory=list)
    vertical_bands: list[BandData] = field(default_factory=list)
    horizontal_lines: list[LineData] = field(default_factory=list)
    horizontal_bands: list[BandData] = field(default_factory=list)
    annotation_arrows: list[ArrowData] = field(default_factory=list)

    # -- mutation -----------------------------------------------------

    def clear_verticals(self) -> None:
        self.vertical_lines.clear()
        self.vertical_bands.clear()

    def clear_horizontals(self) -> None:
        self.horizontal_lines.clear()
        self.horizontal_bands.clear()

    def clear_annotation_arrows(self) -> None:
        self.annotation_arrows.clear()

    # -- rendering ----------------------------------------------------

    def apply(
        self,
        ax: Axes,
        *,
        fig: Figure,
        color_resolver: ColorResolver,
    ) -> None:
        """Render every annotation onto ``ax``.

        ``fig`` is required by the arrow renderer (canvas draw + axes-bbox
        measurements). ``color_resolver`` is passed in so this module does not
        hold a reference back to `GerryPlotBase`.
        """
        self._draw_verticals(ax, color_resolver)
        self._draw_horizontals(ax, color_resolver)
        self._draw_arrows(ax, fig=fig, color_resolver=color_resolver)

    def _draw_verticals(self, ax: Axes, color_resolver: ColorResolver) -> None:
        for band in self.vertical_bands:
            if band.linecolor is None or band.linewidth == 0.0:
                edgecolor = "none"
            else:
                edgecolor = color_resolver(
                    band.linecolor,
                    band.linealpha,
                    field="linecolor",
                )
            ax.axvspan(
                band.lower_bound,
                band.upper_bound,
                facecolor=color_resolver(
                    band.bandcolor,
                    band.bandalpha,
                    field="bandcolor",
                ),
                edgecolor=edgecolor,
                linestyle=band.linestyle,
                linewidth=band.linewidth,
                zorder=band.zorder,
            )

        for line in self.vertical_lines:
            line_values = line.values
            assert isinstance(line_values, Iterable)
            for value in line_values:
                assert isinstance(value, (int, float))
                ax.axvline(
                    value,
                    color=color_resolver(
                        line.linecolor,
                        line.linealpha,
                        field="linecolor",
                    ),
                    linestyle=line.linestyle,
                    linewidth=line.linewidth,
                    zorder=line.zorder,
                )

    def _draw_horizontals(self, ax: Axes, color_resolver: ColorResolver) -> None:
        for band in self.horizontal_bands:
            if band.linecolor is None or band.linewidth == 0.0:
                edgecolor = "none"
            else:
                edgecolor = color_resolver(
                    band.linecolor,
                    band.linealpha,
                    field="linecolor",
                )
            ax.axhspan(
                band.lower_bound,
                band.upper_bound,
                facecolor=color_resolver(
                    band.bandcolor,
                    band.bandalpha,
                    field="bandcolor",
                ),
                edgecolor=edgecolor,
                linestyle=band.linestyle,
                linewidth=band.linewidth,
                zorder=band.zorder,
            )

        for line in self.horizontal_lines:
            line_values = line.values
            assert isinstance(line_values, Iterable)
            for value in line_values:
                assert isinstance(value, (int, float))
                ax.axhline(
                    value,
                    color=color_resolver(
                        line.linecolor,
                        line.linealpha,
                        field="linecolor",
                    ),
                    linestyle=line.linestyle,
                    linewidth=line.linewidth,
                    zorder=line.zorder,
                )

    def _draw_arrows(
        self,
        ax: Axes,
        *,
        fig: Figure,
        color_resolver: ColorResolver,
    ) -> None:
        if not self.annotation_arrows:
            return
        renderer = _AnnotationArrowRenderer(
            ax=ax,
            fig=fig,
            color_resolver=color_resolver,
        )
        renderer.render_all(self.annotation_arrows)
