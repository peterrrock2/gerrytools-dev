from dataclasses import dataclass
from numbers import Integral
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
    shrink: float | None = None  # rarely needed when using cax
    aspect: float | None = None  # rarely needed when using cax

    # --- explicit overrides (optional) ---
    force_ticks: list[float] | None = None
    force_ticklabels: list[str] | None = None
    max_n_ticks: int | None = None

    def __post_init__(self) -> None:
        if self.max_n_ticks is not None:
            if (
                isinstance(self.max_n_ticks, bool)
                or not isinstance(self.max_n_ticks, Integral)
                or self.max_n_ticks < 1
            ):
                raise ValueError("max_n_ticks must be a positive integer.")
            self.max_n_ticks = int(self.max_n_ticks)
        if self.force_ticklabels is not None and self.force_ticks is None:
            raise ValueError("force_ticklabels requires force_ticks.")
        if (
            self.force_ticklabels is not None
            and self.force_ticks is not None
            and len(self.force_ticklabels) != len(self.force_ticks)
        ):
            raise ValueError("force_ticklabels and force_ticks must have the same length.")
