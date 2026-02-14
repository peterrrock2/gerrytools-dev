from dataclasses import dataclass
from typing import Literal


@dataclass(slots=True)
class _ColorbarLayoutOptions:
    """Layout options for positioning colorbars in GeoPlot.
    Attributes:
        outer_pad (float): Padding between the colorbars and the plot edges (figure-relative).
        inner_pad (float): Padding between the colorbars and the main plot area (figure-relative).
        width (float): Width of the colorbars (figure-relative).
        right_margin (float): Margin to the right of the colorbars (figure-relative).
    """

    outer_pad: float = 0.03
    inner_pad: float = 0.06
    width: float = 0.02
    right_margin: float = 0.02


@dataclass(slots=True)
class ColorbarOptions:
    """Options for configuring colorbars in GeoPlot.

    Attributes:
        tick_fontsize (float): Font size for colorbar ticks.
        tick_pad (float): Padding for colorbar ticks.
        label_fontsize (float | None): Font size for colorbar label.
        label_rotation (float | None): Rotation angle for colorbar label.
        label_pad (float | None): Padding for colorbar label.
        orientation (Literal["vertical", "horizontal"]): Orientation of the colorbar.
        extend (Literal["neither", "both", "min", "max"]): Extension style for the colorbar.
        format (str | None): Format string for colorbar tick labels.
        fraction (float | None): Fraction of original size for colorbar.
        shrink (float | None): Shrink factor for colorbar.
        aspect (float | None): Aspect ratio for colorbar.
        force_ticks (list[float] | None): Explicit tick locations for the colorbar.
        force_ticklabels (list[str] | None): Explicit tick labels for the colorbar.
        max_n_ticks (int | None): Maximum number of ticks on the colorbar.
    """

    # --- tick appearance (axes.tick_params) ---
    tick_fontsize: float = 8.0
    tick_pad: float = 2.0

    # --- label appearance (cb.set_label) ---
    label_fontsize: float | None = None
    label_rotation: float | None = None
    label_pad: float | None = None

    # --- fig.colorbar behavior ---
    orientation: Literal["vertical", "horizontal"] = "vertical"
    extend: Literal["neither", "both", "min", "max"] = "neither"
    format: str | None = None  # e.g. ".2f"
    fraction: float | None = None  # rarely needed when using cax
    shrink: float | None = None  # rarely needed when using cax
    aspect: float | None = None  # rarely needed when using cax

    # --- explicit overrides (optional) ---
    force_ticks: list[float] | None = None
    force_ticklabels: list[str] | None = None
    max_n_ticks: int | None = None
