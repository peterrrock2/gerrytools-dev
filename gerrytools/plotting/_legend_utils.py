from __future__ import annotations

from typing import Literal, Sequence

import matplotlib.pyplot as plt

from gerrytools.plotting.mpl.legend_options import LegendOptions
from gerrytools.typing import Color, LegendHandle, MplKwargs


def build_legend_options(
    *,
    loc: str | int = "center left",
    bbox_to_anchor: tuple[float, float] | tuple[float, float, float, float] | None = (1.01, 0.5),
    ncols: int = 1,
    fontsize: float | str | None = None,
    frameon: bool = True,
    fancybox: bool = False,
    shadow: bool = False,
    framealpha: float | None = None,
    facecolor: Color | None = None,
    edgecolor: Color | None = None,
    title: str | None = None,
    alignment: Literal["center", "left", "right"] = "center",
    labelspacing: float = 0.5,
    columnspacing: float = 2.0,
) -> LegendOptions:
    """Build a ``LegendOptions`` instance from standard legend kwargs.

    Args:
        loc (str | int, optional): Legend location passed to Matplotlib. Defaults to
            ``"center left"``.
        bbox_to_anchor (tuple[float, float] | tuple[float, float, float, float] | None, optional):
            Anchor box for legend placement. Defaults to ``(1.01, 0.5)``.
        ncols (int, optional): Number of legend columns. Defaults to ``1``.
        fontsize (float | str | None, optional): Legend font size. Defaults to ``None``.
        frameon (bool, optional): Whether to draw a legend frame. Defaults to ``True``.
        fancybox (bool, optional): Whether to draw a rounded frame. Defaults to ``False``.
        shadow (bool, optional): Whether to draw a shadow. Defaults to ``False``.
        framealpha (float | None, optional): Frame alpha override. Defaults to ``None``.
        facecolor (Color | None, optional): Frame face color. Defaults to ``None``.
        edgecolor (Color | None, optional): Frame edge color. Defaults to ``None``.
        title (str | None, optional): Legend title. Defaults to ``None``.
        alignment (Literal["center", "left", "right"], optional): Text alignment in the
            legend box. Defaults to ``"center"``.
        labelspacing (float, optional): Vertical spacing between entries. Defaults to ``0.5``.
        columnspacing (float, optional): Horizontal spacing between columns. Defaults to ``2.0``.

    Returns:
        LegendOptions: Normalized legend option dataclass.
    """
    return LegendOptions(
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        ncols=ncols,
        fontsize=fontsize,
        frameon=frameon,
        fancybox=fancybox,
        shadow=shadow,
        framealpha=framealpha,
        facecolor=facecolor,
        edgecolor=edgecolor,
        title=title,
        alignment=alignment,
        labelspacing=labelspacing,
        columnspacing=columnspacing,
    )


def save_legend_handles(
    *,
    handles: Sequence[LegendHandle],
    legend_options: LegendOptions,
    filepath: str,
    outer_padding: float = 0.07,
    dpi: int | float = 300,
    **legend_kwargs: object,
) -> None:
    """Save legend handles to a standalone image.

    Args:
        handles (Sequence[LegendHandle]): Handles to render in the legend.
        legend_options (LegendOptions): Base legend options dataclass.
        filepath (str): Output image path.
        outer_padding (float, optional): Fractional expansion applied to the legend bounding box.
            Defaults to ``0.07``.
        dpi (int | float, optional): Save resolution. Defaults to ``300``.
        **legend_kwargs (object): Additional keyword arguments merged into the legend options.

    Raises:
        ValueError: If ``handles`` is empty.
    """
    if len(handles) == 0:
        raise ValueError("No legend handles to save.")

    legend_fig = plt.figure(dpi=dpi)
    try:
        legend_ax = legend_fig.add_subplot(111)
        legend_ax.axis("off")

        opts: MplKwargs = legend_options.to_dict() | dict(legend_kwargs)
        leg = legend_ax.legend(handles=handles, **opts)

        legend_fig.subplots_adjust(0, 0, 1, 1)

        canvas = legend_fig.canvas
        canvas.draw()
        get_renderer_fn = getattr(canvas, "get_renderer", None)
        if not callable(
            get_renderer_fn
        ):  # pragma: no cover - defensive guard for non-standard Matplotlib backends that omit get_renderer()
            raise RuntimeError("Matplotlib canvas does not expose get_renderer().")
        renderer = get_renderer_fn()

        bbox = leg.get_window_extent(renderer=renderer)
        bbox_inches = bbox.transformed(legend_fig.dpi_scale_trans.inverted())
        bbox_inches = bbox_inches.expanded(1.0 + outer_padding, 1.0 + outer_padding)

        legend_fig.savefig(
            filepath,
            bbox_inches=bbox_inches,
            pad_inches=0.0,
        )
    finally:
        plt.close(legend_fig)
