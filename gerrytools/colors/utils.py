from collections.abc import Mapping, Sequence
from typing import Union, cast

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from gerrytools.typing import Color


def preview_palette(
    colors: Sequence[Color],
    figsize: tuple[float, float] = (6, 1),
    show_indices: bool = False,
    show_hex: bool = False,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Preview a color palette as horizontal swatches.

    Args:
        colors: Sequence of colors. Each can be a hex string (e.g. "#ff0000")
            or an RGB triple with components in [0, 1].
        figsize: Size of the figure (width, height). Ignored when ``ax`` is
            provided.
        show_indices: If True, annotate each swatch with its index.
        show_hex: If True, annotate each swatch with its hex code.
        ax: Optional existing axes to draw onto. If None, a fresh figure and
            axes are created.

    Returns:
        The Matplotlib figure and axes.
    """
    # Normalize to RGB triples
    rgb_colors = [mcolors.to_rgb(c) for c in colors]
    n = len(rgb_colors)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = cast(Figure, ax.figure)

    # Draw swatches as vertical bars across the axis
    for i, c in enumerate(rgb_colors):
        ax.bar(
            i,
            1,
            color=c,
            edgecolor="none",
            align="edge",
            width=1.0,
        )

        if show_indices or show_hex:
            text = ""
            if show_indices:
                text += str(i)
            if show_hex:
                if text:
                    text += "\n"
                text += mcolors.to_hex(c)
            ax.text(
                i + 0.5,
                0.5,
                text,
                ha="center",
                va="center",
                fontsize=8,
                color="black" if sum(c) > 1.5 else "white",
            )

    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    return fig, ax


def compare_palettes(
    palettes: Union[Mapping[str, Sequence[Color]], Sequence[Sequence[Color]]],
    figsize: tuple[float, float] | None = None,
    show_hex: bool = False,
) -> tuple[Figure, Axes]:
    """
    Compare multiple color palettes as horizontal rows.

    Args:
        palettes:
            - If dict: {name: [colors...], ...}
            - If list/tuple: [[colors...], [colors...], ...] (rows will be named 0, 1, 2...)
          Colors can be hex strings or RGB triples in [0, 1].
        figsize: Matplotlib figure size (width, height). If None, chosen based
          on number of palettes and max length.
        show_hex: If True, write hex codes inside swatches (can get busy).

    Returns:
        (fig, ax)
    """
    # Normalize palettes to dict[name -> list[RGB triples]]
    items: list[tuple[str, Sequence[Color]]]
    if isinstance(palettes, Mapping):
        # Cast to satisfy the type-checker after runtime Mapping narrowing.
        palette_map = cast(Mapping[str, Sequence[Color]], palettes)
        items = [(str(name), palette) for name, palette in palette_map.items()]
    else:
        items = [(str(i), row) for i, row in enumerate(palettes)]

    norm_palettes: list[tuple[str, list[tuple[float, float, float]]]] = []
    max_len = 0
    for name, colors in items:
        rgb_colors = [mcolors.to_rgb(c) for c in colors]
        norm_palettes.append((name, rgb_colors))
        max_len = max(max_len, len(rgb_colors))

    n_palettes = len(norm_palettes)
    if n_palettes == 0:
        raise ValueError("No palettes provided.")

    # Default figsize: scale with number of colors and palettes
    if figsize is None:
        figsize = (max(4, max_len * 0.5), max(1.5, n_palettes * 0.6))

    fig, ax = plt.subplots(figsize=figsize)

    # Draw each palette as a row (y = row index)
    for row_idx, (name, colors) in enumerate(norm_palettes):
        y = n_palettes - 1 - row_idx  # invert so first is at top
        for col_idx, c in enumerate(colors):
            rect = Rectangle((col_idx, y), 1, 1, facecolor=c, edgecolor="none")
            ax.add_patch(rect)

            if show_hex:
                hex_code = mcolors.to_hex(c)
                ax.text(
                    col_idx + 0.5,
                    y + 0.5,
                    hex_code,
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="black" if sum(c) > 1.5 else "white",
                )

        # Palette label on the left
        ax.text(
            -0.3,
            y + 0.5,
            name,
            ha="right",
            va="center",
            fontsize=9,
        )

    ax.set_xlim(0, max_len)
    ax.set_ylim(0, n_palettes)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    return fig, ax
