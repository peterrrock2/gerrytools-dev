import logging
import math
import re
from numbers import Real
from typing import Any, TypeGuard

import matplotlib.colors as mcolors

from gerrytools.logging import get_logger
from gerrytools.plotting.colors.districtr import DISTRICTR_COLOR_DICT, districtr
from gerrytools.plotting.colors.latex import get_color_from_latex_string
from gerrytools.plotting.colors.latex_full import LATEX_COLOR_DICT
from gerrytools.plotting.colors.seaborn import flare, greens, purples, redbluecmap
from gerrytools.plotting.colors.utils import compare_palettes, preview_palette
from gerrytools.typing import Color

logger = get_logger(__name__)

DEFAULT_GREY = "#5c676f"
"""
Default grey plotting color; used in histograms, violin plots, and arrows.
"""

CITIZEN_BLUE = "#4693b3"
"""
Citizen ensemble blue color; used in histograms, violin plots, and arrows.
"""

OVERLAYS = ("gainsboro", "silver", "darkgray", "gray", "dimgrey")
"""
Overlay colors for choropleth maps.
"""


ENSEMBLE_COLORS = {
    "ensemble:smc": "#ffca5d",
    "ensemble:forest": "#00cd99",
    "ensemble:rrc": "#0099cd",
    "ensemble:revrecom": "#0099cd",
    "ensemble:recoma": "#99cd00",
    "ensemble:recomb": "#cd0099",
    "ensemble:recomc": "#9900cd",
    "ensemble:recomd": "#8dd3c7",
}
"""
A dictionary mapping ensemble abbreviations to their corresponding standard colors.

These were the colors used in the final version of the RRC paper.
"""


GERRYTOOLS_EXTRA_COLORS_DICT = (
    {
        "default_grey": DEFAULT_GREY,
        "default_gray": DEFAULT_GREY,
        "citizen_blue": CITIZEN_BLUE,
    }
    | {name: mcolors.to_hex(mcolors.to_rgba(name, alpha=0.5), keep_alpha=True) for name in OVERLAYS}
    | ENSEMBLE_COLORS
)

HEX8_PATTERN = re.compile(r"^#[0-9A-Fa-f]{8}$")
"""A compiled regular expression pattern to match 8-digit hexadecimal color strings."""
HEX8_OR_NONE_PATTERN = re.compile(r"^(#[0-9A-Fa-f]{8}|none)$", re.IGNORECASE)
"""A compiled regular expression pattern to match 8-digit hexadecimal color strings or "none"."""


def get_all_supported_colors_dict() -> dict[str, Any]:
    """Get a dictionary of all supported color names mapping to their values."""
    return (
        mcolors.get_named_colors_mapping()
        | DISTRICTR_COLOR_DICT
        | LATEX_COLOR_DICT
        | ENSEMBLE_COLORS
        | GERRYTOOLS_EXTRA_COLORS_DICT
        | {"green": "#00ff00"}  # Override matplotlib's dark "green"
        | {"none": "none"}
    )


def get_named_color(name: str) -> Color:
    """Get a color value from the supported color names.

    Args:
        name (str): The name of the color.

    Returns:
        Color: The corresponding color value.

    Raises:
        KeyError: If the color name is not recognized.
    """
    key = name.lower()

    if key == "green":
        return "#00ff00"

    # try “as-is” and lowercased in each mapping
    for d in (
        GERRYTOOLS_EXTRA_COLORS_DICT,
        DISTRICTR_COLOR_DICT,
        LATEX_COLOR_DICT,
        mcolors.get_named_colors_mapping(),
    ):
        if name in d:
            return d[name]  # type: ignore[index]
        if key in d:
            return d[key]  # type: ignore[index]

    raise KeyError(f"Unknown color name: {name!r}")


def _is_real(x: Any) -> TypeGuard[Real]:
    return (
        isinstance(x, Real) and float(x) == float(x) and math.isfinite(float(x))
    )  # filters NaN too


def _is_rgb_tuple(x: Any) -> TypeGuard[tuple[Real, Real, Real]]:
    return isinstance(x, tuple) and len(x) == 3 and all(_is_real(v) for v in x)


def _is_rgba_tuple(x: Any) -> TypeGuard[tuple[Real, Real, Real, Real]]:
    return isinstance(x, tuple) and len(x) == 4 and all(_is_real(v) for v in x)


def _is_mpl_rgb_color(x: Any) -> TypeGuard[str | tuple[Real, Real, Real]]:
    return isinstance(x, str) or _is_rgb_tuple(x)


def _is_mpl_rgba_color(
    x: Any,
) -> TypeGuard[str | tuple[Real, Real, Real, Real] | tuple[str | tuple[Real, Real, Real], Real]]:
    # either a normal rgba tuple, or (base_color, alpha)
    if isinstance(x, str) or _is_rgba_tuple(x):
        return True
    return isinstance(x, tuple) and len(x) == 2 and _is_mpl_rgb_color(x[0]) and _is_real(x[1])


def convert_color_to_hexa_or_none(color: Any) -> str:
    """Convert a color input to a hex8 string or "none".

    Args:
        color (Any): The color input to convert. This can be a named color,
            a LaTeX color string, an RGB(A) tuple, or other formats supported
            by Matplotlib.
    """
    if color is None or (isinstance(color, str) and color.lower() == "none"):
        return "none"

    # ---- strings: named / latex / plain mpl / hex ----
    if isinstance(color, str):
        try:
            c = get_named_color(color)
            return mcolors.to_hex(mcolors.to_rgba(c), keep_alpha=True)
        except KeyError as e:
            logger.debug(f"Color {color!r} is not a known named color string: {e}")

        # LaTeX/xcolor string support
        try:
            rgba = get_color_from_latex_string(color)
            return mcolors.to_hex(rgba, keep_alpha=True)
        except Exception as e:
            logger.debug(f"Color {color!r} is not a LaTeX color string: {e}")

        # Generic matplotlib parsing (also covers '#RRGGBB' and '#RRGGBBAA')
        try:
            return mcolors.to_hex(mcolors.to_rgba(color), keep_alpha=True)
        except Exception as e:
            logger.debug(f"Color {color!r} not parseable by Matplotlib: {e}")

    # ---- (base, alpha) tuples ----
    if (
        isinstance(color, tuple)
        and len(color) == 2
        and _is_mpl_rgb_color(color[0])
        and _is_real(color[1])
    ):
        base, a = color
        rgba = mcolors.to_rgba(base, alpha=float(a))
        return mcolors.to_hex(rgba, keep_alpha=True)

    # ---- numeric RGB(A) tuples (0–1 or 0–255) ----
    if _is_rgb_tuple(color) or _is_rgba_tuple(color):
        vals = [float(v) for v in color]
        r, g, b = vals[:3]
        if max(r, g, b) > 1.0:
            r, g, b = r / 255.0, g / 255.0, b / 255.0
        a = 1.0 if len(vals) == 3 else (vals[3] if vals[3] <= 1.0 else vals[3] / 255.0)
        return mcolors.to_hex((r, g, b, a), keep_alpha=True)

    raise ValueError(f"Unknown color value: {color!r}")


def _validate_alpha(alpha: float, *, field: str = "alpha") -> float:
    """Validate that alpha is convertible to a float and between 0.0 and 1.0."""
    a = float(alpha)
    if not math.isfinite(a):
        raise ValueError(f"{field} must be finite")
    if not (0.0 <= a <= 1.0):
        raise ValueError(f"{field} must be between 0.0 and 1.0")
    return a


def resolve_color_and_alpha(
    color: Any,
    alpha: float | None = None,
    *,
    allow_none: bool = True,
    field: str = "color",
    owner: str | None = None,
    logger: logging.Logger | None = None,
) -> tuple[str, float]:
    """Normalize a (color, alpha) pair into (hex6_or_none, resolved_alpha).

    Rules:
    - color is converted via convert_color_to_hexa_or_none -> "none" or "#RRGGBBAA"
    - if color resolves to "none":
        - allow_none=True  => ("none", 0.0)  (alpha is forced to 0)
        - allow_none=False => ValueError
    - if color resolves to "#RRGGBBAA":
        - returns ("#RRGGBB", alpha_from_color) if alpha is None
        - returns ("#RRGGBB", alpha) if alpha is provided (validated),
          optionally logs when it overrides embedded alpha.


    Args:
        color (Any): The color input to convert.
        alpha (float | None): An optional explicit alpha value between 0.0 and 1.0.

    Kwargs:
        allow_none (bool): Whether "none" is an acceptable color. Defaults to True.
        field (str): The name of the field being processed, for error messages.
        owner (str | None): An optional owner name for logging context.
        logger (logging.Logger | None): An optional logger for debug messages.

    Returns:
        tuple[str, float]: A tuple of (hex6_or_none, resolved_alpha).
    """
    hex8 = convert_color_to_hexa_or_none(color)
    if HEX8_OR_NONE_PATTERN.match(hex8) is None:
        raise ValueError(f"{field} could not be converted to a valid color: {color!r} -> {hex8!r}")

    if hex8.lower() == "none":
        if not allow_none:
            raise ValueError(f"{field} cannot be 'none'.")
        return "none", 0.0

    hex6 = hex8[:7]
    alpha_from_color = int(hex8[7:], 16) / 255.0

    if alpha is None:
        return hex6, alpha_from_color

    a = _validate_alpha(alpha, field=f"{field} alpha")

    if logger is not None and a != alpha_from_color:
        prefix = f"For {owner}: " if owner else ""
        logger.log(
            level=logging.DEBUG,
            msg=(
                f"{prefix}Ignoring alpha embedded in {field} {hex8!r} "
                f"because explicit alpha {a} was provided."
            ),
        )

    return hex6, a


__all__ = [
    "districtr",
    "redbluecmap",
    "flare",
    "purples",
    "greens",
    "convert_color_to_hexa_or_none",
    "get_color_from_latex_string",
    "get_all_supported_colors_dict",
    "DEFAULT_GREY",
    "CITIZEN_BLUE",
    "OVERLAYS",
    "ENSEMBLE_COLORS",
    "compare_palettes",
    "preview_palette",
    "HEX8_PATTERN",
    "HEX8_OR_NONE_PATTERN",
    "resolve_color_and_alpha",
]
